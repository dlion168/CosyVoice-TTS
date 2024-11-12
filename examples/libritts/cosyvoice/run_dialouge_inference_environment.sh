#!/bin/bash
source ~/.bashrc
conda activate cosyvoice

. ./path.sh || exit 1;

pretrained_model_dir=../../../pretrained_models/CosyVoice-300M
data_dir="/nfs/nas-6.1/cjtsao/environmental_sound/DataGeneration/humannoise"
metadata_filename="metadata.json"
audio_subpath="data/{audio_id}"

for mode in zero_shot; do
    python cosyvoice/bin/inference_dialogue_environment.py \
    --mode $mode \
    --gpu 1 \
    --dialogue $1 \
    --config conf/cosyvoice.yaml \
    --prompt_data data/soundon_ref/parquet/data.list \
    --prompt_utt2data data/soundon_ref/parquet/utt2data.list \
    --machine_data data/machine_ref/parquet/data.list \
    --machine_utt2data data/machine_ref/parquet/utt2data.list \
    --llm_model $pretrained_model_dir/../llm_v3.pt \
    --flow_model $pretrained_model_dir/flow.pt \
    --hifigan_model $pretrained_model_dir/hift.pt \
    --result_dir $2 \
    --start_index $3 \
    --num_example $4 \
    --data_dir $data_dir \
    --metadata_filename $metadata_filename \
    --audio_subpath $audio_subpath
done