import os
import json
import random
import argparse
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
import re

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

def split_sentences(long_sentences):
    # Define a list to hold all the shorter sentences
    shorter_sentences = []
    # Define a list to hold the count of shorter sentences for each long sentence
    sentence_counts = []
    
    # Define Chinese punctuation marks that indicate sentence boundaries
    punctuation_marks = r'[。！？]'
    
    for sentence in long_sentences:
        # Split the long sentence based on punctuation marks
        split_result = re.split(punctuation_marks, sentence)
        # Filter out any empty strings from the result and add punctuation back
        split_result = [s + '.' if s else '' for s in re.findall(r'.+?[。！？]', sentence)]
        
        # Count the number of resulting shorter sentences
        count = len(split_result)
        # Add this count to the sentence_counts list
        sentence_counts.append(count)
        # Extend the shorter_sentences list with the split result
        shorter_sentences.extend(split_result)
    
    return shorter_sentences, sentence_counts

def main():
    args = get_args()
    device = get_device(args)
    configs = read_config(args)
    
    random.seed(args.start_index) #1234

    model = load_model(args, configs)
    normalizer = CosyVoiceNormalizer()
    all_possible_references = read_json_lists(args.prompt_utt2data).keys()

    os.makedirs(args.result_dir, exist_ok=True)
    with open(args.dialogue, 'r', encoding='utf-8') as f:
        all_data = f.readlines()
        inference_segment = all_data[args.start_index: args.start_index + args.num_example]

        for index, dialouge in tqdm(enumerate(inference_segment)):
            dialogue_index = index + args.start_index
            if os.path.exists(os.path.join(args.result_dir, f"{dialogue_index:06d}_Machine_meta.json")):
                continue
            
            dialogue = json.loads(dialouge.strip())
            machine_reference = random_sample_reference(all_possible_references)
            ref_mapping = {machine_reference: 'Machine'}
         
            normalized_text = [normalizer.text_normalize(dialogue[key]) for key in dialogue.keys() if key.startswith('Machine_') ]
            tts_sentences, tts_sentence_count = split_sentences(normalized_text)
         
            tts_data = { machine_reference: tts_sentences}
            test_dataset = Dataset(args.prompt_data, data_pipeline=configs['data_pipeline'], mode='inference', shuffle=False, partition=False, tts_data=tts_data, prompt_utt2data=args.prompt_utt2data)
            test_dataloader = DataLoader(test_dataset, batch_size=None, num_workers=0)

            speechs = {'Machine': []}
            accumulated_speeches = []
            sentence_counter = 0
            speech_segment_counter = 0

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
                    accumulated_speeches.append(model_output['tts_speech'])
                    sentence_counter += 1
                    
                    if sentence_counter == tts_sentence_count[speech_segment_counter]:
                        concatenated_speech = torch.concat(accumulated_speeches, dim=1)
                        tts_fn = os.path.join(args.result_dir, f"{dialogue_index:06}_Machine_{speech_segment_counter}.wav")
                        torchaudio.save(tts_fn, concatenated_speech, sample_rate=22050)
                        speechs['Machine'].append(concatenated_speech)
                        
                        accumulated_speeches = []
                        sentence_counter = 0
                        speech_segment_counter += 1

            with open(os.path.join(args.result_dir, f"{dialogue_index:06d}_Machine_meta.json"), "w") as outfile: 
                json.dump({"Machine": machine_reference}, outfile)

if __name__ == "__main__":
    main()
