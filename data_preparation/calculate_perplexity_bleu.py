# This notebook uses https://github.com/openai/whisper with edits to the whisper_openAI/decoding.py to generate multiple hypothesis
import os
from datasets import load_dataset
import tqdm
import argparse
# moving to the whisper folder ; make sure you have the whisper environment on
import numpy
# Renamed the Whisper repo (https://github.com/openai/whisper) with the changed decoding.py file as whisper_openAI
import whisper_openAI.whisper as whisper
from whisper_openAI.whisper.tokenizer import get_tokenizer
from lit_gpt.tokenizer import Tokenizer
from lit_gpt.model import GPT, Block, Config
from lit_gpt.utils import (
    check_valid_checkpoint_dir,
    chunked_cross_entropy,
    load_checkpoint,
    num_parameters,
)
import lightning as L
import torch
import tqdm
from pathlib import Path
import json
import collections

def get_orig_data(args):
    
    if args.dataset == 'fleurs':
        #fleurs
        sdata = load_dataset("google/fleurs", args.language, split=args.data_split, use_auth_token=args.access_token, cache_dir=f'{args.dump_dir}/fleurs')
    elif args.dataset == 'commonvoice':
        #common voice
        language = args.language.split("_")[0] #fleurs format hi_in -> hi for commonvoice
        sdata = load_dataset("mozilla-foundation/common_voice_16_1", language, split=args.data_split, use_auth_token=args.access_token, cache_dir=f'{args.dump_dir}/commonvoice')
    elif args.dataset == 'ramayana':
        ramayana_base_folder = "/dccstor/speech_irl/datasets/speech/ramayana"
        data_folder = os.path.join(ramayana_base_folder, args.data_split, "audio")
        sdata = load_dataset("audiofolder", data_dir=data_folder)["train"]
    
    # Do any data preprocessing here, if any.
    return sdata

def get_final_data(data, args):

    # Setup fabric 
    plugins = None
    if args.quantize is not None and args.quantize.startswith("bnb."):
        if "mixed" in precision:
            raise ValueError("Quantization and mixed precision is not supported.")
        dtype = {"16-true": torch.float16, "bf16-true": torch.bfloat16, "32-true": torch.float32}[precision]
        plugins = BitsandbytesPrecision(args.quantize[4:], dtype)
        precision = None
    
    fabric = L.Fabric(devices=1, precision=args.precision, plugins=plugins)
    fabric.launch()

    fabric.print(args)
    
    print('Loading LLama model....')
    llama_dir: Path = Path(args.llama_model)
    llama_checkpoint_path = llama_dir / "lit_model.pth"
    llama_config = Config.from_name(name=llama_dir.name)

    with fabric.init_module(empty_init=True):
        llama_model = GPT(llama_config)

    load_checkpoint(fabric, llama_model, llama_checkpoint_path, strict=True)
    llama_model = fabric.setup(llama_model)
    
    llama_tokenizer = Tokenizer(llama_dir)
    
    # helper values
    frame_index = list(range(10000))
    
    tot_whisper_perplexity = 0.0    
    tot_llama_perplexity = 0.0
    no_examples = len(data)
    correct_at_position = dict.fromkeys(range(0, 2048), 0)
    total_match_found = 0
    total_lens = 0
    
    count = 0
    output_data = []
    for i in tqdm.tqdm(data):
        ##############################    Data preprocessing    ################################
        # Things to be saved in json
        if args.dataset == 'commonvoice':
            key = i['client_id']
            ground_truth = i['sentence']
            gender = i['gender']
        elif args.dataset == 'fleurs':
            key = i['id']
            ground_truth = i['transcription']
            gender = i['gender']
        elif args.dataset == 'ramayana':
            key = i['unique_id']
            ground_truth = i['transcription']
            gender = key.split("_")[0]
            if not ground_truth:
                total_skipped_empty += 1
                continue

        match_found = 0
        ##############################    Generate llama2 tokens  ################################
        ground_truth_chars = list(ground_truth)
        if '�' in ground_truth:
            continue
        llama_ground_truth_tokens = llama_tokenizer.encode(ground_truth, bos=True, eos=True, max_length=2048)
        llama_tokens_len = len(llama_ground_truth_tokens)
        
        ##############################    Calculate llama perplexities    ################################
        
        # Calculate llama perplexity
        llama_ground_truth_tokens = llama_ground_truth_tokens.to(llama_model.device)
        with torch.no_grad():
            logits = llama_model(llama_ground_truth_tokens.unsqueeze(0))
            logits = logits.squeeze(0)
        
        query_results = []
        ci = 0 # current_index
        while ci < llama_tokens_len:
            if llama_tokenizer.decode(llama_ground_truth_tokens[ci:ci+1]) == '':
                ci += 1
                continue
            l,r = ci,ci+1
            while '�' in llama_tokenizer.decode(llama_ground_truth_tokens[l:r]):
                r += 1
            
            assert '�' not in llama_tokenizer.decode(llama_ground_truth_tokens[l:r])
            ic_logits = logits[l-1:r-1]
            
            _, predicted_tokens = ic_logits.topk(1)
            predicted_char = llama_tokenizer.decode(predicted_tokens.squeeze(-1))
            
            gt_char = llama_tokenizer.decode(llama_ground_truth_tokens[l:r])
            if gt_char in predicted_char:
                match_found += 1
                correct_at_position[l] += 1

            query_results.append((gt_char, predicted_char))
            ci = r
            
        ##############################    Add to json data    ################################
        output_data.append( 
            {
                'index': key,
                'ground_truth':ground_truth, 
                'match_accuracy' : match_found / len(ground_truth_chars),
                'query_results' : ','.join([ f'({entry[0]},{entry[1]})' for entry in query_results])
            } 
        )
        total_match_found += match_found
        total_lens += len(ground_truth_chars) 

        count += 1
        if count % 10 == 0:
            save_json(output_data, args)
    
    sorted_position = collections.OrderedDict(sorted(correct_at_position.items()))
    output_data.append(
        {
            'Average Overall Accuracy': total_match_found / total_lens,
            'correct_at_position' : sorted_position
        }
    )
    return output_data

def save_json(data, args):
    save_dir = f'{args.out_dir}/'
    llama_base = (args.llama_model.split("/")[-1])
    filename = f'{args.dataset}_{args.language}_whisper_{args.model_size}_{llama_base}_{args.data_split}.json'
    save_path =f'{save_dir}/{filename}'
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    with open(save_path,'w') as f:
        f.write(json.dumps(data, indent=4, ensure_ascii=False))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script to extract encoder and decoder representations from whisper",add_help=False)
    parser.add_argument("--dump-dir","-ed",type=str,required=True,help="Path for experiment folder file.")
    parser.add_argument("--out-dir","-od",type=str,required=True,help="Path for experiment folder file.")
    parser.add_argument("--access-token","-at",type=str,required=True,help="Huggingface access token to download dataset.")
    parser.add_argument("--dataset","-dt",type=str,required=True,default='fleurs',help="The name of the dataset.")
    parser.add_argument('--precision', type=str, default=None, help='TBD(default: None)') 
    parser.add_argument('--quantize', type=str, default=None, choices=["bnb.nf4", "bnb.nf4-dq", "bnb.fp4", "bnb.fp4-dq", "bnb.int8-training"], help='TBD(default: None)') 
    parser.add_argument("--data-split","-ds",type=str,required=True,choices=['train', 'validation', 'test'],help="The name of the data split. Should be one of: ['train', 'validation', 'test']")
    parser.add_argument("--model-size","-ms",type=str,default="tiny",choices=['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3'], help="Whisper model type. Should be one of: ['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3']")
    parser.add_argument("--llama_model","-lm",type=str,default="meta-llama/Llama-2-7b-hf", help="Valid llama model.")
    parser.add_argument("--language","-ln",type=str,default="hi_in", help="Langauge for the dataset.")
    args = parser.parse_args()
    
    print(args)
    category2data = get_orig_data(args)
    data = get_final_data(category2data, args)
    save_json(data, args)