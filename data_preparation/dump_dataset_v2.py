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
import torch
import tqdm
from pathlib import Path

def detokenize(tokenizer, ids):
    print([tokenizer.decode(ids[x:x+1]) for x in range(len(ids))])

def get_orig_data(args):
    
    if args.dataset == 'fleurs':
        #fleurs
        sdata = load_dataset("google/fleurs", args.language, split=args.data_split, use_auth_token=args.access_token, cache_dir=f'{args.dump_dir}/fleurs')
    elif args.dataset == 'commonvoice':
        #common voice
        language = args.language.split("_")[0] #fleurs format hi_in -> hi for commonvoice
        sdata = load_dataset("mozilla-foundation/common_voice_16_1", language, split=args.data_split, use_auth_token=args.access_token, cache_dir=f'{args.dump_dir}/commonvoice')
    
    # Do any data preprocessing here, if any.
    return sdata

def get_final_data(data, args):

    model = whisper.load_model(args.model_size) #for vanilla model
    model.eval()

    whisper_language = args.language.split("_")[0]
    
    # Variable to hold intermediate logits for interCTC loss calculation
    intermediate_logits = []
    
    whisper_tokenizer = get_tokenizer(
        model.is_multilingual,
        num_languages=model.num_languages,
        language=whisper_language,
        task="transcribe",
    )
    
    llama_tokenizer_path: Path = Path(args.llama_model)
    llama_tokenizer = Tokenizer(llama_tokenizer_path)
    
    output_data = []
    total_skipped_length = 0
    total_skipped_empty = 0
    for i in tqdm.tqdm(data):
        assert len(intermediate_logits) == 0
        
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
        
        path_to_file = i['audio']['path']

        audio_duration = len(i['audio']['array']) / i['audio']['sampling_rate']

        ##############################    Generate llama2 tokenization and state alignments  ################################
        llama_ground_truth_tokens = llama_tokenizer.encode(ground_truth, bos=True, eos=True, max_length=2048)
        input_ids = llama_ground_truth_tokens.clone()
        labels = llama_ground_truth_tokens.clone()
        
        llama_token_len = llama_ground_truth_tokens.size(0)
        state_mappings = torch.tensor([0]*llama_token_len)

        previous_llama_ground_truth = ''
        if args.debug:
            print(ground_truth)
        whisper_ground_truth_tokens = list(whisper_tokenizer.sot_sequence) + [whisper_tokenizer.no_timestamps]

        whisper_index = 3
        state_mappings[0] = 3
        for token_index in range(1, llama_token_len):
            #first llama token is <s> and the last token is </s>
            current_llama_ground_truth = llama_tokenizer.decode(llama_ground_truth_tokens[:token_index + 1])
            if ground_truth.startswith(current_llama_ground_truth) and current_llama_ground_truth != previous_llama_ground_truth:
                if args.debug:
                    print(f"Llama Chunks@{token_index}: {current_llama_ground_truth} ||| {list(current_llama_ground_truth[len(previous_llama_ground_truth):])} ")
                
                current_word_to_tokenize = current_llama_ground_truth[len(previous_llama_ground_truth):]

                current_whisper_tokens = whisper_tokenizer.encode(current_word_to_tokenize)
                whisper_ground_truth_tokens += current_whisper_tokens
                previous_llama_ground_truth = current_llama_ground_truth

                whisper_index += len(current_whisper_tokens)
                state_mappings[token_index] = whisper_index
            else:
                state_mappings[token_index] = whisper_index

        original_whisper_ground_truth_tokens = list(whisper_tokenizer.sot_sequence) + [whisper_tokenizer.no_timestamps] + whisper_tokenizer.encode(ground_truth)
        assert whisper_tokenizer.decode(original_whisper_ground_truth_tokens) == whisper_tokenizer.decode(whisper_ground_truth_tokens), f"Tokenization mismatch found! {key}"
        
        whisper_token_len = len(whisper_ground_truth_tokens)
        if args.data_split != 'test' and whisper_token_len > 448:
            total_skipped_length += 1
            if args.debug:
                print(f"Skipping long utterance {key} with length {whisper_token_len}")
            continue
        elif args.data_split != 'test' and 'Llama-2-13b-hf' in args.llama_model and llama_token_len > 448:
            total_skipped_length += 1
            #if args.debug:
            print(f"Skipped Long utterance: {key} with length {llama_token_len}")
            continue
        
        def pretty_print():
            from tabulate import tabulate
            tabulate.PRESERVE_WHITESPACE = True
            table = [['.']*(whisper_token_len-1) for _ in range(llama_token_len)]
            
            whisper_ids_to_tokens = ['-'] + ['<SOT>'] + [whisper_tokenizer.decode([x]) for x in whisper_ground_truth_tokens][4:]
            llama_ids_to_tokens = [llama_tokenizer.decode(torch.tensor([x])) for x in llama_ground_truth_tokens]        

            for ind, state in enumerate(state_mappings):
                table[ind][0] = llama_ids_to_tokens[ind]
                table[ind][state-1] = 'X'
            
            table = [whisper_ids_to_tokens] + table
            
            table_to_display = [x[:25] for x in table[:25]]
            print(tabulate(table_to_display, headers='firstrow', tablefmt='fancy_grid', colalign=("left")))
        
        if args.debug:
            pretty_print()

        encoder_state = None
        decoder_states = None
        
        if not args.lazy_dump:
            #During lazy dump we avoid computation of whisper features to have tractable files and postpone the computation to the finetune method.
            
            # Prepare audio input
            audio =  i['audio']['array'].astype(numpy.single)         
            audio = whisper.pad_or_trim(audio)
            mel = whisper.log_mel_spectrogram(audio, model.dims.n_mels).to(model.device) 
            mel = mel.unsqueeze(0)
            
            ##############################    Extract encoder and decoder representations    ################################
            if args.data_split != 'test':
                whisper_ground_truth_tokens = torch.tensor(whisper_ground_truth_tokens).to(mel.device)
                whisper_ground_truth_tokens = whisper_ground_truth_tokens.unsqueeze(0)
                
                def get_intermediate_output(module, input, output):
                    intermediate_logits.append(output[0])

                # Register hooks for all the encoder layers we intend to tap into.
                hooks = []
                for layer in model.decoder.blocks:
                    hook = layer.register_forward_hook(get_intermediate_output)
                    hooks.append(hook)

                with torch.no_grad():
                    encoder_state  = model.encoder(mel)
                    _ = model.logits(whisper_ground_truth_tokens, encoder_state)
                
                encoder_state = encoder_state.squeeze(0).detach().cpu().float()
                decoder_states = torch.stack(intermediate_logits, dim=0).detach().cpu().float()
                
                # Remove all hooks
                for hook in hooks:
                    hook.remove()
                
                # Flush out the old intermediate logits
                intermediate_logits = []
            
            else:
                whisper_ground_truth_tokens = None
                with torch.no_grad():
                    encoder_state  = model.encoder(mel)
                encoder_state = encoder_state.squeeze(0).detach().cpu().float()
            
        # pretty_print()
        ##############################    Add to dataset    ################################
        output_data.append( 
            {
                'index': key,
                'input_ids': input_ids,
                'path':path_to_file ,
                'ground_truth':ground_truth, 
                'gender':gender, 
                'audio_duration': audio_duration,
                'encoder_states': encoder_state,
                'decoder_states': decoder_states,
                'labels': labels,
                'state_mappings': state_mappings,
                'whisper_ground_truth_tokens': whisper_ground_truth_tokens 
            } 
        )

    print(f"Skipped due to long duration:  {total_skipped_length}")
    print(f"Skipped due to empty transcript:  {total_skipped_empty}")
    return output_data

def save_torch(data, args):
    save_dir = f'{args.dump_dir}/'
    llama_base = (args.llama_model.split("/")[-1])
    filename = f'{args.dataset}_{args.language}_whisper_{args.model_size}_{llama_base}_{args.data_split}.pt'
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    save_path =f'{save_dir}/{filename}'
    torch.save(data, save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script to extract encoder and decoder representations from whisper",add_help=False)
    parser.add_argument("--dump-dir","-ed",type=str,required=True,help="Path for experiment folder file.")
    parser.add_argument("--access-token","-at",type=str,required=True,help="Huggingface access token to download dataset.")
    parser.add_argument("--dataset","-dt",type=str,required=True,default='fleurs',help="The name of the dataset.")
    parser.add_argument("--data-split","-ds",type=str,required=True,choices=['train', 'validation', 'test'],help="The name of the data split. Should be one of: ['train', 'validation', 'test']")
    parser.add_argument("--model-size","-ms",type=str,default="tiny",choices=['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3'], help="Whisper model type. Should be one of: ['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3']")
    parser.add_argument("--llama_model","-lm",type=str,default="meta-llama/Llama-2-7b-hf", help="Valid llama model.")
    parser.add_argument("--debug","-db",type=bool,default=False, help="To print debugging statements.")
    parser.add_argument("--language","-ln",type=str,default="hi_in", help="Langauge for the dataset.")
    parser.add_argument("--lazy_dump","-lz",type=bool,default=False,help="Lazy dump set to True avoids dumping of Whisper features for large datasets.")
    args = parser.parse_args()    
    
    print(args)
    category2data = get_orig_data(args)
    data = get_final_data(category2data, args)
    save_torch(data, args)