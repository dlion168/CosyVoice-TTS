import os
import json
import random
import argparse
import re
from tqdm import tqdm
from functools import partial
from hyperpyyaml import load_hyperpyyaml

import torch
import torchaudio
from torch.utils.data import DataLoader

from cosyvoice.cli.model import CosyVoiceModel
from cosyvoice.dataset.dataset import DataList, Processor
from cosyvoice.utils.file_utils import read_lists, read_json_lists
from cosyvoice.utils.normalize import CosyVoiceNormalizer
from cosyvoice.utils.environment_sound import separate_text_and_sound, combine_speech_and_environment

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

def merge_speechs(speechs, sil_duration=0.2, sr=22050):
    sil_tensor = torch.zeros(1, int(sil_duration * sr))
    merged_speeches = []
    for u_speech, m_speech in zip(speechs['User'], speechs['Machine']):
        merged_speeches.append(u_speech)
        merged_speeches.append(sil_tensor)
        merged_speeches.append(m_speech)
        merged_speeches.append(sil_tensor)
    return torch.concat(merged_speeches[:-1], dim=1)


def main():
    args = get_args()
    device = get_device(args)
    configs = read_config(args)
    normalizer = CosyVoiceNormalizer()

    model = load_model(args, configs)
    all_possible_references = read_json_lists(args.prompt_utt2data).keys()

    os.makedirs(args.result_dir, exist_ok=True)
    with open(args.dialogue, 'r', encoding='utf-8') as f:
        all_data = f.readlines()
        inference_segment = all_data[args.start_index: args.start_index + args.num_example]

        for index, dialouge in tqdm(enumerate(inference_segment)):
            dialogue = json.loads(dialouge.strip())
            user_reference = random_sample_reference(all_possible_references)
            machine_reference = random_sample_reference(all_possible_references)
            ref_mapping = {user_reference: 'User', machine_reference: 'Machine'}

            environment_sounds = []
            for dialogue_key in dialogue.keys():
                if dialogue_key.startswith('User_'):
                    dialogue[dialogue_key], environment_sound = separate_text_and_sound(dialogue[dialogue_key])
                    environment_sounds.append(environment_sound)

            tts_data = {
                user_reference: [
                    normalizer.text_normalize(dialogue[key]) for key in dialogue.keys() if key.startswith('User_')
                ],
                machine_reference: [
                    normalizer.text_normalize(dialogue[key]) for key in dialogue.keys() if key.startswith('Machine_')
                ]
            }


            test_dataset = Dataset(args.prompt_data, data_pipeline=configs['data_pipeline'], mode='inference', shuffle=False, partition=False, tts_data=tts_data, prompt_utt2data=args.prompt_utt2data)
            test_dataloader = DataLoader(test_dataset, batch_size=None, num_workers=0)

            speechs = {
                'User': [],
                'Machine': []
            }

            dialogue_index = index + args.start_index
            merge_tts_key = '{:06d}_all'.format(dialogue_index)
            merge_tts_fn = os.path.join(args.result_dir, '{}.wav'.format(merge_tts_key))
            if os.path.exists(merge_tts_fn):
                continue

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

                    tts_key = '{:06d}_{}_{}'.format(dialogue_index, ref_mapping[utts[0]], tts_index[0])
                    tts_fn = os.path.join(args.result_dir, '{}.wav'.format(tts_key))
                    if os.path.exists(tts_fn):
                        continue

                    tts_speeches = []
                    # print(model.inference(**model_input))
                    # for model_output in model.inference(**model_input):
                    #     tts_speeches.append(model_output['tts_speech'])
                    model_output = model.inference(**model_input)
                    tts_speeches.append(model_output['tts_speech'])
                    speechs[ref_mapping[utts[0]]].append(model_output['tts_speech'])

                    # print("utts: {}".format(utts[0]))
                    # print("tts_index: {}".format(tts_index[0]))
                    tts_speeches = torch.concat(tts_speeches, dim=1)
                    if ref_mapping[utts[0]].startswith('User'):
                        tts_speeches = combine_speech_and_environment(tts_speeches, environment_sounds[tts_index[0]])
                    torchaudio.save(tts_fn, tts_speeches, sample_rate=22050)

            # tts_key = '{:06d}_all'.format(dialogue_index)
            # tts_fn = os.path.join(args.result_dir, '{}.wav'.format(tts_key))
            # if len(speechs['User']) != 7 or len(speechs['Machine']) != 7:
            #     user_files = [os.path.join(args.result_dir, '{:06d}_User_{}.wav'.format(dialogue_index, i)) for i in range(7)]
            #     machine_files = [os.path.join(args.result_dir, '{:06d}_Machine_{}.wav'.format(dialogue_index, i)) for i in range(7)]
            #     speechs = {
            #         'User': [torchaudio.load(f)[0] for f in user_files],
            #         'Machine': [torchaudio.load(f)[0] for f in machine_files]
            #     }
            # merged_speeches = merge_speechs(speechs)
            # torchaudio.save(tts_fn, merged_speeches, sample_rate=22050)

if __name__ == "__main__":
    main()
