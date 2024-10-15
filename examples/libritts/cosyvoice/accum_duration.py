import os
import librosa
import argparse
from tqdm import tqdm

def get_total_duration(dir):
  total_duration = 0
  flist = [f for f in os.listdir(dir) if f.endswith(args.ext)]
  for f in tqdm(flist):
    fpath = os.path.join(dir, f)
    wav, sr = librosa.load(fpath, sr=None)
    total_duration += librosa.get_duration(y=wav, sr=sr)
  return total_duration

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--src_dir',
                      type=str)
  parser.add_argument('--ext', 
                      type=str)
  args = parser.parse_args()
  total_duration = get_total_duration(args.src_dir)
  print(f"Total duration of all audio files: {total_duration} seconds")