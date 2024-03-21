# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import sys, os
from pathlib import Path

import lightning as L
import torch

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from generate.whispering_adapter import generate
from lit_gpt import Tokenizer
from lit_gpt.whispering_adapter_v1 import GPT, Config
from lit_gpt.utils import check_valid_checkpoint_dir, lazy_load
import whisper_openAI.whisper as whisper
from whisper_openAI.whisper.tokenizer import get_tokenizer

import argparse
from tqdm import tqdm
import json
import pdb
from datasets import load_dataset
import numpy


from evaluate import load
wer = load("wer")
cer = load("cer")
ter = load("ter")

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', type=Path, required=True, help='Path to the finetuned adapter model checkpoint') 
    parser.add_argument("--dump-dir","-ed",type=str,required=True,help="Path for experiment folder file.")
    parser.add_argument('--output-file', type=str, required=True, help='Filename to save the inference results') 
    parser.add_argument("--access-token","-at",type=str,required=True,help="Huggingface access token to download dataset.")
    parser.add_argument("--whisper-model",type=str,default="tiny",choices=['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3'], help="Whisper model type. Should be one of: ['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3'] (default: 'tiny')")
    parser.add_argument("--dataset","-dt",type=str,required=True,default='fleurs',help="The name of the dataset.")
    parser.add_argument("--data-split","-ds",type=str,required=True,choices=['train', 'validation', 'test'],help="The name of the data split. Should be one of: ['train', 'validation', 'test']")
    parser.add_argument('--top-k', type=int, default=1, help='TBD(default: 1)') 
    parser.add_argument('--temperature', type=float, default=0.0, help='TBD(default: 0.0)') 
    parser.add_argument("--language","-ln",type=str,default="hi_in", help="Langauge for the dataset.")
    parser.add_argument("--resume_inference","-rn",type=bool,default="False", help="Resume the inference from the last decoded state.")
    parser.add_argument("--lazy_dump","-lz",type=bool,default=False, help="Lazy dump set to True create whisper features during inference.")    
    
    return parser

def setup():
    # Parse arguements
    parser = get_parser()
    args = parser.parse_args()
    
    return args, None

def main() -> None:
    args, _ = setup()
    print(args)
    
    whisper_model = whisper.load_model(args.whisper_model)
    whisper_model.eval()
    
    # Variable to hold intermediate logits for interCTC loss calculation
    intermediate_logits = []
    
    completely_accurate = 0
    predictions = []
    ground_truths = []
    whisper_language = args.language.split("_")[0]

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # Saving the results to a JSON file here
    inference_path = args.out_dir / args.output_file
    if os.path.exists(inference_path) and args.resume_inference:
        with open(inference_path, 'r', encoding='utf-8') as f:
            to_json = json.load(f)

        #computation already done, need to repeat. Just Print and return
        if to_json and "processed cer" in to_json[-1]:
            print("All the results are computed in previous run. Sharing it here:")
            unprocessed_results = to_json[-2]
            print('*********************')
            print(f"Unprocessed CER is {unprocessed_results['unprocessed cer']}")
            print(f"Unprocessed WER is {unprocessed_results['unprocessed wer']}")
            print(f"Unprocessed TER is {unprocessed_results['unprocessed ter']}")
            print(f"Unprocessed Ground truth matches is {unprocessed_results['unprocessed accurate predictions']}")
            print('*********************')

            processed_results = to_json[-1]
            print('*********************')
            print(f"Processed CER is {processed_results['processed cer']}")
            print(f"Processed WER is {processed_results['processed wer']}")
            print(f"Processed TER is {processed_results['processed ter']}")
            print(f"Processed Ground truth matches is {processed_results['processed accurate predictions']}")
            print('*********************')
            return
        else:
            #Read the previous predictions so that the final WER computation is correct.
            for json_row in to_json:
                predictions.append(json_row['inference'])
                ground_truths.append(json_row['ground_truth'])
                if json_row['inference'] == json_row['ground_truth']:
                    completely_accurate += 1
    else:
        to_json = []

    if args.lazy_dump:
        from datasets import Dataset, Audio
    
    if args.dataset == 'fleurs':
        sdata = load_dataset("google/fleurs", args.language, split=args.data_split, use_auth_token=args.access_token, cache_dir=f'{args.dump_dir}/fleurs')
    
    for index, data in enumerate(tqdm(sdata)):
        
        key = data['id']
        ground_truth = data['transcription']
        
        audio = data['audio']['array'].astype(numpy.single)  
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio, whisper_model.dims.n_mels).to(whisper_model.device) 
        mel = mel.unsqueeze(0)
        options = whisper.DecodingOptions(without_timestamps=True, language=whisper_language)
        inference = whisper.decode(whisper_model, mel, options)
        output = inference[0].text
        
        reference = ground_truth.strip()
        
        if output == reference: # we increase the count if the inference compleately matches the ground
            completely_accurate += 1
        predictions.append(output)
        ground_truths.append(reference)
        to_json.append({'id':key,'inference':output, 'ground_truth':reference})
        print(f"Ground Truth: {reference}")
        print(f"Prediction: {output}")
        with open(inference_path,'w') as f:
            f.write(json.dumps(to_json , indent=4, ensure_ascii=False))
        sys.stdout.flush()
    
    cer_ = cer.compute(predictions=predictions, references=ground_truths)
    wer_ = wer.compute(predictions=predictions, references=ground_truths)
    #ter_ = ter.compute(predictions=predictions, references=ground_truths)
    print('*********************')
    print(f'Unprocessed CER is {cer_}')
    print(f'Unprocessed WER is {wer_}')
    #print(f'Unprocessed TER is {ter_}')
    print(f'Unprocessed Ground truth matches is {completely_accurate}/{len(data)}')
    print('*********************')
    #to_json.append({'unprocessed cer':cer_, 'unprocessed wer':wer_, 'unprocessed ter':ter_, 'unprocessed accurate predictions':f'{completely_accurate}/{len(data)}'})
    to_json.append({'unprocessed cer':cer_, 'unprocessed wer':wer_, 'unprocessed accurate predictions':f'{completely_accurate}/{len(data)}'})

    completely_accurate = 0    
    for i in range(len(predictions)):
        predictions[i] = predictions[i].lower().translate(str.maketrans({c:None for c in "।.,-?':"}))
        ground_truths[i] = ground_truths[i].lower().translate(str.maketrans({c:None for c in "।.,-?':"}))
        if predictions[i] == ground_truths[i]:
            completely_accurate += 1
    
    cer_ = cer.compute(predictions=predictions, references=ground_truths)
    wer_ = wer.compute(predictions=predictions, references=ground_truths)
    #ter_ = ter.compute(predictions=predictions, references=ground_truths)
    print('*********************')
    print(f'Processed CER is {cer_}')
    print(f'Processed WER is {wer_}')
    #print(f'Processed TER is {ter_}')
    print('*********************')
    print(f'Processed Ground truth matches is {completely_accurate}/{len(sdata)}')
    #to_json.append({'processed cer':cer_, 'processed wer':wer_, 'processed ter':ter_, 'processed accurate predictions':f'{completely_accurate}/{len(data)}'})
    to_json.append({'processed cer':cer_, 'processed wer':wer_, 'processed accurate predictions':f'{completely_accurate}/{len(sdata)}'})
    
    with open(inference_path,'w') as f:
        f.write(json.dumps(to_json , indent=4, ensure_ascii=False))


if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")
    main()
