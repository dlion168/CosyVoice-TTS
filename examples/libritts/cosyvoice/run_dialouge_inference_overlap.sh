#!/bin/bash
# source ~/.bashrc
# conda activate cosyvoice

# Usage:
# First, modify ml2021_reference and soundon_ref pointed to desire datasets
# Then, modify the inference_target. Supporting "user", "machine" or "both"
# bash run_dialouge_inference_overlap.sh /path/to/tts_text.jsonl /path/to/result_dir start_inference_index_in_tts_text.jsonl(int)
# (You can add new arg --num_example $4 if needed, it will inference {num_example} dialogues starts from jsonl line indexed $3)

. ./path.sh || exit 1;

pretrained_model_dir=../../../pretrained_models/CosyVoice-300M

for mode in zero_shot; do
    for user_data in ml2021_reference; do
        for machine_data in soundon_ref; do
            echo "mode: zero_shot"
            python cosyvoice/bin/inference_dialogue_overlap.py \
            --mode $mode \
            --gpu 0 \
            --dialogue $1 \
            --config conf/cosyvoice.yaml \
            --prompt_data_user data/$user_data/parquet/data.list \
            --prompt_data_machine data/$machine_data/parquet/data.list \
            --prompt_utt2data_user data/$user_data/parquet/utt2data.list \
            --prompt_utt2data_machine data/$machine_data/parquet/utt2data.list \
            --llm_model $pretrained_model_dir/../llm_v2.pt \
            --flow_model $pretrained_model_dir/flow.pt \
            --hifigan_model $pretrained_model_dir/hift.pt \
            --result_dir $2 \
            --start_index $3 \
            --num_example $4 \
            --inference_target user
        done
    done
done
