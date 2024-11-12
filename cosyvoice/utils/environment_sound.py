import re
import json
import random
import wave
import numpy as np
import torch
import torchaudio


data_dir = "/nfs/nas-6.1/cjtsao/environmental_sound/DataGeneration/fsd50k"
with open(f"{data_dir}/label_id.json", "r") as f:
    label_to_ids = json.load(f)


def parse_tag(tag):
    tag = tag[1:-1]
    pattern = r"([^\s]+)(\s+[\w]+=[^>]+)*"
    match = re.match(pattern, tag)
    
    if match:
        parsed_dict = {"name": match.group(1)}
        attributes = re.findall(r"(\w+)=([^\s>]+)", tag)
        for key, value in attributes:
            parsed_dict[key] = float(value) if value.replace('.', '', 1).isdigit() else value
        return parsed_dict
    return None

def audio_normalize(audio, volume=1):
    denominator = max(abs(audio.max()), abs(audio.min()))
    audio = audio / denominator * volume
    return audio

def generate_environmental_sound(tag):
    config = parse_tag(tag)
    sound_list = label_to_ids.get(config["name"], [])
    if not sound_list:
        raise ValueError(f"Sound not found for tag: {config['name']}")
    audio_id = random.choice(sound_list)
    filepath = f"{data_dir}/FSD50K.dev_audio/{audio_id}.wav"
    waveform, sample_rate = torchaudio.load(filepath)
    new_sample_rate = 22050
    resampler = torchaudio.transforms.Resample(sample_rate, new_sample_rate)
    waveform = resampler(waveform)
    audio_tensor = audio_normalize(waveform, volume=config.get("volume", 1))
    time_limit = 3
    audio_tensor = audio_tensor[:, :new_sample_rate * time_limit]
    return audio_tensor

def extract_environment_tag(text, matches):
    total_tag_length = 0
    for i, match in enumerate(matches):
        matches[i] = {
            "tag": match[0],
            "type": "interleave" if match[0][0] == "[" else "background",
            "position": (match[1] - total_tag_length) / len(text),
            "audio": generate_environmental_sound(match[0])
        }
        print(matches[i]["audio"].size())
        total_tag_length += len(match[0])
    return matches

def find_tags(text):
    pattern = r"(\[.*?\]|\<.*?\>)"
    matches = [(match.group(), match.start(), match.end()) for match in re.finditer(pattern, text)]
    text = re.sub(pattern, '', text)
    return text, matches

def separate_text_and_sound(text):
    text, matches = find_tags(text)
    tags = extract_environment_tag(text, matches)
    return text, tags

def combine_speech_and_environment(speech, tags):
    # TODO multiple tags
    # Remove batch dimension to work directly with the 1D audio signal
    tag = tags[0]
    speech = speech[0]
    tag_audio = tag["audio"][0]
    
    # Calculate the insertion position
    position = int(speech.size(0) * tag["position"])
    
    if tag["type"] == "interleave":
        # Interleave operation: Insert audio at specified position
        speech = torch.cat((speech[:position], tag_audio, speech[position:]))
    
    else:  # "background" type
        # Calculate the end position for overlaying background audio
        end = position + tag_audio.size(0)
        
        # Pad speech if needed
        if end > speech.size(0):
            padding = torch.zeros(end - speech.size(0))
            speech = torch.cat((speech, padding))
        
        # Add the background audio directly to the specified range
        speech[position:end] += tag_audio
    
    # Return the modified speech tensor with the batch dimension restored
    return speech.unsqueeze(0)

# Example usage

if __name__ == "__main__":
    text, tags = separate_text_and_sound("<Boiling volume=0.1> 有沒有聽到什麼聲音？")

    print("Text after removal:", text)
    print("Tags with indices:", tags)

    example_audio, sample_rate = torchaudio.load("/nfs/nas-6.1/cjtsao/environmental_sound/CosyVoice-TTS/examples/libritts/cosyvoice/test_dir/000000_User_0.wav")
    combined_audio = combine_speech_and_environment(example_audio, tags[0])
    torchaudio.save("combined_audio.wav", combined_audio, sample_rate=22050)