#!/usr/bin/env bash

# export CUDA_VISIBLE_DEVICES=6

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

# General configuration
stage=0                                           # Processes starts from the specified stage.
stop_stage=10000
llama_model=Llama-2-7b-hf                         # Processes is stopped at the specified stage.
llama_repo_id=meta-llama/Llama-2-7b-hf            # LLama model saved at this huggingface repo is used as base model
hf_access_token=                                  # This access token is used to download the model from huggingface
whisper_model=large-v2                            # Whisper model to be used for audio features
model_save_dir=models                             # Downloaded models are saved in this directory
dump_dir=dataset                                  # Generated datasets are saved here
dataset=fleurs                                    # Name of the dataset
exp_dir=exp                                       # Experiment checkpoints and necessary files are stored here
language=hi_in                                    # Language for the experiment.

help_message=$(cat << EOF
Usage: $0 --train-set --hf_access_token "<valid_set_name>" --test_sets "<test_set_names>"

Options:
    # General configuration
    --stage                # Processes starts from the specified stage (default="${stage}").
    --stop_stage           # Processes is stopped at the specified stage (default="${stop_stage}").
    --llama_repo_id        # LLama model saved at this huggingface repo is used as base model (default="${llama_repo_id}").
    --hf_access_token      # This access token is used to download the model from huggingface (required).
    --whisper_model        # Whisper model to be used for audio features. Should be one of: ['tiny', 'base', 'small', 'medium', 'large', 'large-v2'] (default="${whisper_model}").
    --model_save_dir       # Downloaded models are saved in this directory (default="${model_save_dir}").
    --dump_dir             # Generated datasets are saved here (default="${dump_dir}").
    --exp_dir              # Experiment checkpoints and necessary files are stored here (default="${dump_dir}").
    
EOF
)

log "$0 $*"
# Save command line args for logging (they will be lost after utils/parse_options.sh)
run_args=$(utils/print_args.sh $0 "$@")
. utils/parse_options.sh

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "Stage 0: Download LLama model and convert it to required format"

    if [[ -z "${hf_access_token}" ]]; then
        log "${help_message}"; 
        log "Error: --hf_access_token is required"; 
        exit 2;
    fi

    python scripts/download.py \
        --repo_id ${llama_repo_id} \
        --access_token ${hf_access_token} \
        --checkpoint_dir ${model_save_dir}

    python scripts/convert_hf_checkpoint.py --checkpoint_dir ${model_save_dir}/${llama_repo_id}
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Stage 1: Create train, validation and test dataset"
    mkdir -p ${dump_dir}
    
    if [[ -z "${hf_access_token}" ]]; then
        log "${help_message}"; 
        log "Error: --hf_access_token is required"; 
        exit 2;
    fi

    for split in train validation test; do
        python data_preparation/dump_dataset_v2.py \
            --dump-dir ${dump_dir} \
            --access-token ${hf_access_token}  \
            --data-split ${split} \
            --dataset ${dataset} \
            --model-size ${whisper_model}  \
            --llama_model ${model_save_dir}/${llama_repo_id} \
            --language ${language}
    done;
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "Stage 2: Finetune LLama model"
    mkdir -p ${exp_dir}
    
    python finetune/finetune.py \
        --train-data-path ${dump_dir}/${dataset}_${language}_whisper_${whisper_model}_${llama_model}_train.pt \
        --val-data-path ${dump_dir}/${dataset}_${language}_whisper_${whisper_model}_${llama_model}_validation.pt \
        --model-save-dir ${model_save_dir}/${llama_repo_id} \
        --whisper-model ${whisper_model} \
        --exp-dir ${exp_dir}/${dataset}_${language}_whisper_${whisper_model}_${llama_model} \
        --downsampling-factor 4
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "Stage 3: Compute the length model"
    mkdir -p ${exp_dir}
    
    python data_preparation/calculate_length_model.py \
        --train-data-path ${dump_dir}/${dataset}_${language}_whisper_${whisper_model}_${llama_model}_train.pt \
        --val-data-path ${dump_dir}/${dataset}_${language}_whisper_${whisper_model}_${llama_model}_validation.pt \
        --exp-dir ${exp_dir}/${dataset}_${language}_whisper_${whisper_model}_${llama_model}
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    log "Stage 4: Inference on the finetuned LLama model"

    for split in test; do
        python inference/inference.py \
            --checkpoint-dir ${model_save_dir}/${llama_repo_id} \
            --exp-dir ${exp_dir}/${dataset}_${language}_whisper_${whisper_model}_${llama_model} \
            --whisper-model ${whisper_model} \
            --language ${language} \
            --max-new-tokens 500 \
            --data-path ${dump_dir}/${dataset}_${language}_whisper_${whisper_model}_${llama_model}_${split}.pt \
            --output-file results_${language}_${split}.json \
            --downsampling-factor 4 \
            --use-nucleus-sampling True \
            --use-length-model True
    done;
fi


if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    log "Stage 5: Whisper Inference"

    for split in test; do
        python inference/inference_whisper.py \
            --dump-dir ${dump_dir} \
            --access-token ${hf_access_token}  \
            --out-dir ${exp_dir}/whisper_baseline_results/${whisper_model} \
            --dataset ${dataset} \
            --data-split ${split} \
            --whisper-model ${whisper_model} \
            --language ${language} \
            --output-file results_${dataset}_${language}_${split}.json \
            --resume_inference True 
    done;
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    log "Stage 6: Finetune Multilingual LLama model"
    mkdir -p ${exp_dir}
    
    python finetune/finetune_multilingual.py \
        --train-data-path ${dump_dir}/${dataset}_{LNH}_whisper_${whisper_model}_${llama_model}_train.pt \
        --val-data-path ${dump_dir}/${dataset}_{LNH}_whisper_${whisper_model}_${llama_model}_validation.pt \
        --model-save-dir ${model_save_dir}/${llama_repo_id} \
        --whisper-model ${whisper_model} \
        --num-epochs 35 \
        --exp-dir ${exp_dir}/${dataset}_whisper_${whisper_model}_${llama_model}_multilingual \
        --downsampling-factor 4
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    log "Stage 7: Inference on the finetuned Multilingual LLama model"
    for split in validation test; do
        python inference/inference.py \
            --checkpoint-dir ${model_save_dir}/${llama_repo_id} \
            --exp-dir ${exp_dir}/${dataset}_whisper_${whisper_model}_${llama_model}_multilingual \
            --whisper-model ${whisper_model} \
            --language ${language} \
            --max-new-tokens 500 \
            --data-path ${dump_dir}/${dataset}_${language}_whisper_${whisper_model}_${llama_model}_${split}.pt \
            --output-file results_${language}_${split}.json \
            --downsampling-factor 4 \
            --use-nucleus-sampling True \
            --use-length-model True 
    done;
fi
