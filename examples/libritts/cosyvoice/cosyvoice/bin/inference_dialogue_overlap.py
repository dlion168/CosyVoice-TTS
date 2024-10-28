# Output format: /path/to/result/dir/{dialogue_id}/{dialogue_id}_{role}_{sentence_id}_{sentence_segment_id}


import re
import os
import json
import random
import argparse
from tqdm import tqdm
from functools import partial
from hyperpyyaml import load_hyperpyyaml
from typing import Any, Dict, List, Tuple

import torch
import torchaudio
from torch.utils.data import DataLoader

from cosyvoice.cli.model import CosyVoiceModel
from cosyvoice.dataset.dataset import DataList, Processor
from cosyvoice.utils.file_utils import read_lists, read_json_lists

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='inference with your model')
    parser.add_argument('--dialogue', required=True, help='dialogue file')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--prompt_data_user', required=True, help='prompt data file for user')
    parser.add_argument('--prompt_data_machine', required=True, help='prompt data file for machine')
    parser.add_argument('--prompt_utt2data_user', required=True, help='prompt utt2data file for user')
    parser.add_argument('--prompt_utt2data_machine', required=True, help='prompt utt2data file for machine')
    parser.add_argument('--llm_model', required=True, help='llm model file')
    parser.add_argument('--flow_model', required=True, help='flow model file')
    parser.add_argument('--hifigan_model', required=True, help='hifigan model file')
    parser.add_argument('--start_index', required=True, type=int, default=0)
    parser.add_argument('--num_example', type=int)
    parser.add_argument('--gpu', type=int, default=-1, help='gpu id for this rank, -1 for cpu')
    parser.add_argument('--mode', default='zero_shot', choices=['sft', 'zero_shot'], help='inference mode')
    parser.add_argument('--result_dir', required=True, help='asr result file')
    parser.add_argument('--inference_target', required=True, choices=['user', 'machine', 'both'], help='inference only user, machine or both')
    args = parser.parse_args()
    print(args)
    return args

def get_device(args) -> torch.device:
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    return torch.device('cuda' if use_cuda else 'cpu')

def read_config(args) -> dict:
    with open(args.config, 'r') as f:
        configs = load_hyperpyyaml(f)
    return configs

def load_model(args, configs):
    model = CosyVoiceModel(configs['llm'], configs['flow'], configs['hift'])
    model.load(args.llm_model, args.flow_model, args.hifigan_model)
    return model

def Dataset(data_list_file, data_pipeline, mode, shuffle=True, partition=True, tts_data='', prompt_utt2data=''):
    assert mode in ['train', 'inference']
    lists = read_lists(data_list_file)
    if mode == 'inference':
        utt2lists = read_json_lists(prompt_utt2data)
        lists = list({utt2lists[utt] for utt in tts_data.keys() if utt2lists[utt] in lists})
    dataset = DataList(lists, shuffle=shuffle, partition=partition)
    if mode == 'inference':
        data_pipeline[0] = partial(data_pipeline[0], tts_data=tts_data)
    for func in data_pipeline:
        dataset = Processor(dataset, func, mode=mode)
    return dataset

def random_sample_reference(all_possible_references):
    return random.choice(list(all_possible_references))

def assign_roles(main_role: str, text: str, sentence_index: int) -> List[Tuple[int, str, str]]:
    parts = re.split(r'(\(.*?\))', text)
    side_role = 'Machine' if main_role == 'User' else 'User'
    result = []
    for part in parts:
        if part.startswith('(') and part.endswith(')'):
            result.append((sentence_index, side_role, part[1:-1]))
        else:
            if part.strip():
                result.append((sentence_index, main_role, part.strip()))
    return result

def text_to_speech(ref_speaker, texts, model, args, configs, device, role):
    tts_data = {ref_speaker: texts}
    prompt_data_file = args.prompt_data_user if role == "User" else args.prompt_data_machine
    prompt_utt2data_file = args.prompt_utt2data_user if role == "User" else args.prompt_utt2data_machine
    test_dataset = Dataset(prompt_data_file, data_pipeline=configs['data_pipeline'], mode='inference', shuffle=False, partition=False, tts_data=tts_data, prompt_utt2data=prompt_utt2data_file)
    test_dataloader = DataLoader(test_dataset, batch_size=None, num_workers=0)
    
    with torch.no_grad():
        for _, batch in enumerate(test_dataloader):
            utts = batch["utts"]
            assert len(utts) == 1, "inference mode only supports batchsize 1"
            text_token = batch["text_token"].to(device)
            text_token_len = batch["text_token_len"].to(device)
            tts_text_token = batch["tts_text_token"].to(device)
            tts_text_token_len = batch["tts_text_token_len"].to(device)
            speech_token = batch["speech_token"].to(device)
            speech_token_len = batch["speech_token_len"].to(device)
            speech_feat = batch["speech_feat"].to(device)
            speech_feat_len = batch["speech_feat_len"].to(device)
            utt_embedding = batch["utt_embedding"].to(device)
            spk_embedding = batch["spk_embedding"].to(device)
            model_input = {
                'text': tts_text_token,
                'text_len': tts_text_token_len,
                'prompt_text': text_token, 
                'prompt_text_len': text_token_len,
                'llm_prompt_speech_token': speech_token, 
                'llm_prompt_speech_token_len': speech_token_len,
                'flow_prompt_speech_token': speech_token, 
                'flow_prompt_speech_token_len': speech_token_len,
                'prompt_speech_feat': speech_feat, 
                'prompt_speech_feat_len': speech_feat_len,
                'llm_embedding': utt_embedding, 
                'flow_embedding': utt_embedding
            }
            model_output = model.inference(**model_input)
            tts_speech = [model_output['tts_speech']]
            tts_speech = torch.concat(tts_speech, dim=1)
    return tts_speech

def main():
    args = get_args()
    device = get_device(args)
    configs = read_config(args)
    cosyvoice = load_model(args, configs)

    os.makedirs(args.result_dir, exist_ok=True)
    with open(args.dialogue, 'r', encoding='utf-8') as f:
        all_data = f.readlines()
        all_possible_references_user = read_json_lists(args.prompt_utt2data_user).keys()
        all_possible_references_machine = read_json_lists(args.prompt_utt2data_machine).keys()
        inference_segment = all_data[args.start_index: args.start_index + args.num_example] if args.num_example else all_data[args.start_index:]
        
        for index, dialogue in tqdm(enumerate(inference_segment), total=len(inference_segment)):
            dialogue = json.loads(dialogue.strip())
            dialogue_index = args.start_index + index

            user_reference = random_sample_reference(all_possible_references_user)
            machine_reference = random_sample_reference(all_possible_references_machine)
            ref_mapping = {'User': user_reference, 'Machine': machine_reference}

            user_messages = [dialogue[key] for key in dialogue if key.startswith("User")]
            bot_messages = [dialogue[key] for key in dialogue if key.startswith("Machine")]

            if len(user_messages) == 0 or len(bot_messages) == 0:
                continue

            messages = []
            for index, key in enumerate(dialogue):
                if key.startswith("User"):
                    user_msg = dialogue[key]
                    messages += assign_roles(main_role='User', text=user_msg, sentence_index=index)
                elif key.startswith("Machine"):
                    bot_msg = dialogue[key]
                    messages += assign_roles(main_role='Machine', text=bot_msg, sentence_index=index)

            current_sentence_number = -1
            successive_count = 0  

            for (sentence_number, role, msg) in messages:
                if sentence_number == current_sentence_number:
                    successive_count += 1
                else:
                    current_sentence_number = sentence_number
                    successive_count = 0
                tts_key = f'{dialogue_index}_{role.lower()}_{sentence_number}_{successive_count}' 
                
                dialogue_output_dir = os.path.join(args.result_dir, f"{dialogue_index}")
                os.makedirs(dialogue_output_dir, exist_ok=True)
                tts_fn = os.path.join(dialogue_output_dir, f'{tts_key}.wav')
                
                if (role=="User" and (args.inference_target=="user" or args.inference_target=="both")) or (role=="Machine" and (args.inference_target=="machine" or args.inference_target=="both")):
                    tts_speech = text_to_speech(
                        ref_speaker=ref_mapping[role],
                        texts=[msg],
                        model=cosyvoice,
                        args=args,
                        configs=configs,
                        device=device,
                        role=role
                    )
                    torchaudio.save(tts_fn, tts_speech, sample_rate=22050)

if __name__ == "__main__":
    main()
