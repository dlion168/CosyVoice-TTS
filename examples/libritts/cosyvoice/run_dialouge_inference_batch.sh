#!/bin/bash
# source ~/.bashrc
# conda activate cosyvoice

. ./path.sh || exit 1;

pretrained_model_dir=../../../pretrained_models/CosyVoice-300M

for mode in $1; do
    for x in ml2021_reference; do
        echo "mode: $1"
        python cosyvoice/bin/inference_dialogue_batch.py \
        --mode $mode \
        --gpu 0 \
        --dialogue $2 \
        --config conf/cosyvoice.yaml \
        --prompt_data data/$x/parquet/data.list \
        --prompt_utt2data data/$x/parquet/utt2data.list \
        --llm_model $pretrained_model_dir/../llm.pt \
        --flow_model $pretrained_model_dir/flow.pt \
        --hifigan_model $pretrained_model_dir/hift.pt \
        --result_dir $3 \
        --start_index $4 \
	--num_example $5
    done
done
