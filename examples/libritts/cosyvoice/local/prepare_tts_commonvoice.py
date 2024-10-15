import logging
import json
import glob
import os
from tqdm import tqdm
from datasets import load_dataset
import random
import pandas as pd

def random_sample_reference(wavs):
    return os.path.basename(random.sample(wavs, 1)[0])

def main():
    random.seed(33)
    ref_wavs = pd.read_csv('./data/ml2021_reference/wav.scp', sep=' ', header=None)[0].tolist()
    ds = load_dataset("mozilla-foundation/common_voice_16_1", "zh-TW", split="test")
    
    data = {}
    for example in tqdm(ds): 
        ref = random_sample_reference(ref_wavs)
        if ref in data:
            data[ref].append(example['sentence'])
        else:
            data[ref] = [example['sentence']]
    
    with open('tts_text_commonvoice_zhtw.json', 'w') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()