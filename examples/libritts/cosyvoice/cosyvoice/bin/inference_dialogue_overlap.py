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
    parser.add_argument('--prompt_data', required=True, help='prompt data file')
    parser.add_argument('--prompt_utt2data', required=True, help='prompt data file')
    parser.add_argument('--llm_model', required=True, help='llm model file')
    parser.add_argument('--flow_model', required=True, help='flow model file')
    parser.add_argument('--hifigan_model', required=True, help='hifigan model file')
    parser.add_argument('--start_index', required=True, type=int)
    parser.add_argument('--num_example', type=int, default=200)
    parser.add_argument('--gpu',
                        type=int,
                        default=-1,
                        help='gpu id for this rank, -1 for cpu')
    parser.add_argument('--mode',
                        default='zero_shot',
                        choices=['sft', 'zero_shot'],
                        help='inference mode')
    parser.add_argument('--result_dir', required=True, help='asr result file')
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

def load_model(
    args, 
    configs
):
    model = CosyVoiceModel(configs['llm'], configs['flow'], configs['hift'])
    model.load(args.llm_model, args.flow_model, args.hifigan_model)
    return model

def Dataset(
    data_list_file,
    data_pipeline,
    mode='train',
    shuffle=True,
    partition=True,
    tts_data='',
    prompt_utt2data=''
):
    assert mode in ['train', 'inference']
    lists = read_lists(data_list_file)
    if mode == 'inference':
        utt2lists = read_json_lists(prompt_utt2data)
        # filter unnecessary file in inference mode
        lists = list({utt2lists[utt] for utt in tts_data.keys() if utt2lists[utt] in lists})
    dataset = DataList(lists,
                       shuffle=shuffle,
                       partition=partition)
    if mode == 'inference':
        # map partial arg tts_data in inference mode
        data_pipeline[0] = partial(data_pipeline[0], tts_data=tts_data)
    for func in data_pipeline:
        dataset = Processor(dataset, func, mode=mode)
    return dataset

def random_sample_reference(all_possible_references):
    return random.choice(list(all_possible_references))

# def merge_speechs(speechs, sil_duration=0.2, sr=22050):
#     sil_tensor = torch.zeros(1, int(sil_duration * sr))
#     merged_speeches = []
#     for u_speech, m_speech in zip(speechs['User'], speechs['Machine']):
#         merged_speeches.append(u_speech)
#         merged_speeches.append(sil_tensor)
#         merged_speeches.append(m_speech)
#         merged_speeches.append(sil_tensor)
#     return torch.concat(merged_speeches[:-1], dim=1)

def assign_roles(
    main_role: str,
    text: str
) -> List[Tuple[str, str]]:
    """
    Splits the input text into parts based on parentheses and assigns roles
    alternately between the main role for outside parentheses content and 
    another role (machine) for content inside parentheses.

    Args:
        main_role (str): The role to assign for content outside parentheses.
        text (str): The text to process.

    Returns:
        List[Tuple[str, str]]: A list of tuples where each tuple contains:
            - The role (either main_role or "machine").
            - The corresponding part of the text.
    """
    # Split the text by parentheses, keeping the parentheses content
    parts = re.split(r'(\(.*?\))', text)
    side_role = 'Machine' if main_role == 'User' else 'User'
    
    result = []
    
    for part in parts:
        if part.startswith('(') and part.endswith(')'):
            # Content inside parentheses, assign to 'machine'
            result.append((side_role, part[1:-1]))  # Remove parentheses
        else:
            # Content outside parentheses, assign to main_role
            if part.strip():  # Avoid adding empty strings
                result.append((main_role, part.strip()))
    
    return result    

def merge_consecutive_tuples(conversations: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """
    Merges consecutive tuples with the same role by concatenating their messages.

    Args:
        conversations (List[Tuple[str, str]]): A list of tuples where each tuple contains:
            - A role (e.g., 'user' or 'machine').
            - A message string.

    Returns:
        List[Tuple[str, str]]: A new list where consecutive tuples with the same role are merged,
        with their messages concatenated, separated by a blank space.
    """
    if not conversations:
        return []
    
    merged_conversations = [conversations[0]]  # Start with the first conversation
    
    for role, message in conversations[1:]:
        last_role, last_message = merged_conversations[-1]
        
        if role == last_role:
            # If the current role is the same as the last role, merge the messages
            merged_conversations[-1] = (role, last_message + ' ' + message)
        else:
            # Otherwise, just append the new tuple
            merged_conversations.append((role, message))
    
    return merged_conversations

def text_to_speech(
    ref_speaker: str,
    texts: List[str],
    model: CosyVoiceModel,
    args: Dict,
    configs,
    device,
):
    tts_data = {ref_speaker: texts}
    test_dataset = Dataset(args.prompt_data, data_pipeline=configs['data_pipeline'], mode='inference', shuffle=False, partition=False, tts_data=tts_data, prompt_utt2data=args.prompt_utt2data)
    test_dataloader = DataLoader(test_dataset, batch_size=None, num_workers=0)

    with torch.no_grad():
        for _, batch in enumerate(test_dataloader):
            utts = batch["utts"]
            assert len(utts) == 1, "inference mode only support batchsize 1"
            text_token = batch["text_token"].to(device)
            text_token_len = batch["text_token_len"].to(device)
            tts_index = batch["tts_index"]
            tts_text_token = batch["tts_text_token"].to(device)
            tts_text_token_len = batch["tts_text_token_len"].to(device)
            speech_token = batch["speech_token"].to(device)
            speech_token_len = batch["speech_token_len"].to(device)
            speech_feat = batch["speech_feat"].to(device)
            speech_feat_len = batch["speech_feat_len"].to(device)
            utt_embedding = batch["utt_embedding"].to(device)
            spk_embedding = batch["spk_embedding"].to(device)
            if args.mode == 'sft':
                model_input = {'text': tts_text_token, 'text_len': tts_text_token_len,
                            'llm_embedding': spk_embedding, 'flow_embedding': spk_embedding}
            else:
                model_input = {'text': tts_text_token, 'text_len': tts_text_token_len,
                            'prompt_text': text_token, 'prompt_text_len': text_token_len,
                            'llm_prompt_speech_token': speech_token, 'llm_prompt_speech_token_len': speech_token_len,
                            'flow_prompt_speech_token': speech_token, 'flow_prompt_speech_token_len': speech_token_len,
                            'prompt_speech_feat': speech_feat, 'prompt_speech_feat_len': speech_feat_len,
                            'llm_embedding': utt_embedding, 'flow_embedding': utt_embedding}
        
            model_output = model.inference(**model_input)
            tts_speech = [model_output['tts_speech']]
            tts_speech = torch.concat(tts_speech, dim=1)
    return tts_speech

def main():
    args = get_args()
    device = get_device(args)
    configs = read_config(args)

    cosyvoice = load_model(args, configs)
    all_possible_references = read_json_lists(args.prompt_utt2data).keys()

    os.makedirs(args.result_dir, exist_ok=True)
    with open(args.dialogue, 'r', encoding='utf-8') as f:
        all_data = f.readlines()
        inference_segment = all_data[args.start_index: args.start_index + args.num_example]

        for index, dialouge in tqdm(enumerate(inference_segment)):
            dialogue = json.loads(dialouge.strip())
            dialogue_index = args.start_index + index

            user_reference = random_sample_reference(all_possible_references)
            machine_reference = random_sample_reference(all_possible_references)
            ref_mapping = {
                'User': user_reference,
                'Machine': machine_reference
            }
            # ref_mapping = {user_reference: 'User', machine_reference: 'Machine'}

            user_messages = [dialogue[key] for key in dialogue if key.startswith("User")]
            bot_messages = [dialogue[key] for key in dialogue if key.startswith("Machine")]

            if len(user_messages) == 0 or len(bot_messages) == 0:
                continue

            messages = []
            for user_msg, bot_msg in zip(user_messages, bot_messages):
                messages += assign_roles(main_role='User', text=user_msg)
                messages += assign_roles(main_role='Machine', text=bot_msg)

            merge_messages = merge_consecutive_tuples(messages)
            for index, (role, msg) in enumerate(merge_messages):
                tts_key = '{:06d}_{}_{}'.format(dialogue_index, role, index//2)
                tts_fn = os.path.join( args.result_dir, '{}.wav'.format(tts_key) )
                # if os.path.exists(tts_fn):
                #     continue

                # print("{}:\n{}\n".format(role, msg))
                tts_speech = text_to_speech(
                    ref_speaker=ref_mapping[role],
                    texts=[msg],
                    model=cosyvoice,
                    args=args,
                    configs=configs,
                    device=device
                )
                torchaudio.save(tts_fn, tts_speech, sample_rate=22050)

if __name__ == "__main__":
    main()