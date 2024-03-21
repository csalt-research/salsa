# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import sys, os, math
from pathlib import Path

import lightning as L
import torch
from lightning.fabric.plugins import BitsandbytesPrecision

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
import numpy as np

from evaluate import load
wer = load("wer")
cer = load("cer")
ter = load("ter")

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-dir', type=Path, default='checkpoints/stabilityai/stablelm-base-alpha-3b', help='Path to directory hosting Llama2 checkpoint') 
    parser.add_argument('--exp-dir', type=Path, required=True, help='Path to the finetuned adapter model checkpoint') 
    parser.add_argument('--output-file', type=str, required=True, help='Filename to save the inference results') 
    parser.add_argument("--whisper-model",type=str,default="tiny",choices=['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3'], help="Whisper model type. Should be one of: ['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3'] (default: 'tiny')")
    parser.add_argument('--data-path', type=Path, required=True, help='Path to the .pt file on which inference must be performed') 
    parser.add_argument('--downsampling-factor', type=int, default=1, help='factor by which input is downsampled before feeding to llama (default: 1)') 
    parser.add_argument('--normalize-before', type=bool, default=False, help='If true, whisper representation is normalized before passing to llama (default: False)') 
    parser.add_argument('--precision', type=str, default=None, help='TBD(default: None)') 
    parser.add_argument('--quantize', type=str, default=None, choices=["bnb.nf4", "bnb.nf4-dq", "bnb.fp4", "bnb.fp4-dq", "bnb.int8-training"], help='TBD(default: None)') 
    parser.add_argument('--max-new-tokens', type=int, default=300, help='TBD(default: 300)') 
    parser.add_argument('--max-seq-length', type=int, default=2048, help='TBD(default: 2048)') 
    parser.add_argument('--top-k', type=int, default=5, help='TBD(default: 1)') 
    parser.add_argument('--devices', '-d', type=int, default=1, help='No of GPUs (default: 1)')
    parser.add_argument('--top-p', type=float, default=0.9, help='TBD(default: 1)') 
    parser.add_argument('--temperature', type=float, default=0.0, help='TBD(default: 0.0)') 
    parser.add_argument("--language","-ln",type=str,default="hi_in", help="Langauge for the dataset.")
    parser.add_argument("--lazy_dump","-lz",type=bool,default=False, help="Lazy dump set to True create whisper features during inference.")    
    parser.add_argument("--use-nucleus-sampling",type=bool,default=False, help="Use nucleus sampling instead of greedy decoding.")    
    parser.add_argument("--use-length-model",type=bool,default=False, help="Use Length Model.")   
    
    return parser

def setup():
    # Parse arguements
    parser = get_parser()
    args = parser.parse_args()
    args.resume_inference = False
    
    # Load datasets
    data = torch.load(args.data_path, map_location=torch.device('cpu'))
    
    check_valid_checkpoint_dir(args.checkpoint_dir)
    
    # Setup fabric 
    plugins = None
    if args.quantize is not None and args.quantize.startswith("bnb."):
        if "mixed" in precision:
            raise ValueError("Quantization and mixed precision is not supported.")
        dtype = {"16-true": torch.float16, "bf16-true": torch.bfloat16, "32-true": torch.float32}[precision]
        plugins = BitsandbytesPrecision(args.quantize[4:], dtype)
        precision = None
    
    fabric = L.Fabric(devices=args.devices, precision=args.precision, plugins=plugins)
    fabric.launch()

    fabric.print(args)
    
    return args, fabric, data

def main() -> None:
    args, fabric, data = setup()
    
    config = Config.from_json(args.checkpoint_dir / "lit_config.json")
    whisper_model = whisper.load_model(args.whisper_model)

    whisper_model.eval()
    config.whisper_dim = whisper_model.decoder.ln.normalized_shape[0]
    config.no_whisper_decoder_layers = len(whisper_model.decoder.blocks)
    config.downsampling_factor = args.downsampling_factor 
    config.normalize_before = args.normalize_before 
    
    whisper_language = args.language.split("_")[0]
    
    # Variable to hold intermediate logits for interCTC loss calculation
    intermediate_logits = []
    
    whisper_tokenizer = get_tokenizer(
        whisper_model.is_multilingual,
        num_languages=whisper_model.num_languages,
        language=whisper_language,
        task="transcribe",
    )
    
    if args.use_length_model:
        import joblib
        from sklearn.preprocessing import PolynomialFeatures

        poly_model_file = args.exp_dir / "poly_feature_3"
        poly_features = joblib.load(poly_model_file)
        
        length_model_file = args.exp_dir / "length_model_degree_3"
        length_model = joblib.load(length_model_file)
        print(f"Using length model: {length_model_file}")

    checkpoint_path = args.checkpoint_dir / "lit_model.pth"
    adapter_path = args.exp_dir / "lit_model_adapter_finetuned.pth"

    llama_tokenizer = Tokenizer(args.checkpoint_dir)
    
    with fabric.init_module(empty_init=True):
        llama_model = GPT(config)
    with fabric.init_tensor():
        # set the max_seq_length to limit the memory usage to what we need
        llama_model.max_seq_length = args.max_seq_length
        # enable the kv cache
        llama_model.set_kv_cache(batch_size=1)
    llama_model.eval()
    
    checkpoint = lazy_load(checkpoint_path)
    adapter_checkpoint = lazy_load(adapter_path)
    checkpoint.update(adapter_checkpoint.get("model", adapter_checkpoint))
    llama_model.load_state_dict(checkpoint)

    llama_model = fabric.setup(llama_model)
    
    completely_accurate = 0
    predictions = []
    ground_truths = []

    # Saving the results to a JSON file here
    inference_path = args.exp_dir / args.output_file
    to_json = []

    if args.lazy_dump:
        import numpy
        from datasets import Dataset, Audio
    for index, datapoint in enumerate(tqdm(data)):

        if args.lazy_dump:

            audio_path = datapoint["path"]

            audio_dataset = Dataset.from_dict({"audio": [audio_path]}).cast_column("audio", Audio())

            #get features from the audio path.
            audio_array = audio_dataset[0]['audio']['array'].astype(numpy.single)
            audio_array = whisper.pad_or_trim(audio_array)

            mel = whisper.log_mel_spectrogram(audio_array, whisper_model.dims.n_mels)
            mel = mel.unsqueeze(0).to(whisper_model.device)

            with torch.no_grad():
                encoder_states  = whisper_model.encoder(mel).squeeze(0)
        else:
            encoder_states = datapoint['encoder_states'].to(llama_model.device).to(llama_model.dtype)

        print(f"Index: {datapoint['index']}")

        if not args.use_length_model:
            max_returned_tokens = args.max_new_tokens
        else:
            audio_duration = np.array([datapoint['audio_duration']]).reshape(-1, 1)
            audio_duration_poly = poly_features.transform(audio_duration)
            max_returned_tokens = math.ceil(length_model.predict(audio_duration_poly)[0]) + 50 #since regression is not perfect, err on having longer max tokens.
            print(f"Max Returned Tokens from the length model: {max_returned_tokens}")

        ground_truth =  datapoint['ground_truth']
        print(f"Groundtruth: {ground_truth}")
        
        if args.use_nucleus_sampling:
            with torch.no_grad():
                y, sentence_ended = generate(llama_model, llama_tokenizer, whisper_model, whisper_tokenizer, encoder_states, max_returned_tokens, temperature=args.temperature, top_k=args.top_k, top_p=args.top_p, eos_id=llama_tokenizer.eos_id)
        else:
            with torch.no_grad():
                y, sentence_ended = generate(llama_model, llama_tokenizer, whisper_model, whisper_tokenizer, encoder_states, max_returned_tokens, temperature=args.temperature, eos_id=llama_tokenizer.eos_id)
                
        for block in llama_model.transformer.h:
            block.attn.kv_cache.reset_parameters()
        
        output = llama_tokenizer.decode(y).replace('�','')
        reference = ground_truth.strip()
        
        if output == reference: # we increase the count if the inference compleately matches the ground
            completely_accurate += 1
        predictions.append(output)
        ground_truths.append(reference)
        to_json.append({'id':datapoint['index'],'inference':output, 'ground_truth':reference})
        print(f"Prediction: {output}")
        with open(inference_path,'w') as f:
            f.write(json.dumps(to_json , indent=4, ensure_ascii=False))
        sys.stdout.flush()
    
    cer_ = cer.compute(predictions=predictions, references=ground_truths)
    wer_ = wer.compute(predictions=predictions, references=ground_truths)
    ter_ = ter.compute(predictions=predictions, references=ground_truths)
    print('*********************')
    print(f'Unprocessed CER is {cer_}')
    print(f'Unprocessed WER is {wer_}')
    print(f'Unprocessed TER is {ter_}')
    print(f'Unprocessed Ground truth matches is {completely_accurate}/{len(data)}')
    to_json.append({'unprocessed cer':cer_, 'unprocessed wer':wer_, 'unprocessed ter':ter_, 'unprocessed accurate predictions':f'{completely_accurate}/{len(data)}'})
    print('*********************')

    completely_accurate = 0    
    for i in range(len(predictions)):
        predictions[i] = predictions[i].lower().translate(str.maketrans({c:None for c in ".,-?':"}))
        ground_truths[i] = ground_truths[i].lower().translate(str.maketrans({c:None for c in ".,-?':"}))
        if predictions[i] == ground_truths[i]:
            completely_accurate += 1
    
    cer_ = cer.compute(predictions=predictions, references=ground_truths)
    wer_ = wer.compute(predictions=predictions, references=ground_truths)
    ter_ = ter.compute(predictions=predictions, references=ground_truths)
    print('*********************')
    print(f'Processed CER is {cer_}')
    print(f'Processed WER is {wer_}')
    print(f'Processed TER is {ter_}')
    print(f'Processed Ground truth matches is {completely_accurate}/{len(data)}')
    to_json.append({'processed cer':cer_, 'processed wer':wer_, 'processed ter':ter_, 'processed accurate predictions':f'{completely_accurate}/{len(data)}'})
    
    print('*********************')
    with open(inference_path,'w') as f:
        f.write(json.dumps(to_json , indent=4, ensure_ascii=False))


if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")
    main()
