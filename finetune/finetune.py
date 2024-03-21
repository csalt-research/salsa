# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import lightning as L
import torch
from lightning.fabric.loggers import CSVLogger
from lightning.fabric.plugins import BitsandbytesPrecision
from lightning.fabric.strategies import FSDPStrategy
from lightning.fabric.utilities import ThroughputMonitor

from tqdm import tqdm
import numpy
from datasets import Dataset, Audio

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import whisper_openAI.whisper as whisper

from lit_gpt.whispering_adapter_v1 import GPT, Block, Config, adapter_filter, mark_only_adapter_as_trainable
from lit_gpt.utils import (
    check_valid_checkpoint_dir,
    chunked_cross_entropy,
    load_checkpoint,
    num_parameters,
)
import argparse
from argparse import Namespace
from lit_gpt.tokenizer import Tokenizer

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-save-dir', type=Path, default='checkpoints/stabilityai/stablelm-base-alpha-3b', help='Path to directory hosting Llama2 checkpoint') 
    parser.add_argument('--train-data-path', type=Path, default="data/alpaca/train.pt", help='Path to the .pt file containing training data') 
    parser.add_argument('--val-data-path', type=Path, default="data/alpaca/val.pt", help='Path to the .pt file containing validation data') 
    parser.add_argument('--data-dir', type=Path, default="data/alpaca", help='Path to directory containing `train.pt` and `test.pt`') 
    parser.add_argument('--exp-dir', type=Path, default="out/adapter/alpaca", help='Path to directory hosting experiment logs and checkpoints') 
    parser.add_argument('--learning-rate','-lr', type=float, default=1e-3, help='learning rate for the model finetuning (default: 1e-3)')
    parser.add_argument('--weight-decay', type=float, default=0.02, help='TBD (default: 0.02)')
    parser.add_argument('--downsampling-factor', type=int, default=1, help='factor by which input is downsampled before feeding to llama (default: 1)') 
    parser.add_argument('--no-layers-per-adapter', type=int, default=1, help='factor by which input is downsampled before feeding to llama (default: 1)') 
    parser.add_argument('--normalize-before', type=bool, default=False, help='If true, whisper representation is normalized before passing to llama (default: False)') 
    parser.add_argument('--devices', '-d', type=int, default=1, help='No of GPUs (default: 1)')
    parser.add_argument("--whisper-model",type=str,default="tiny",choices=['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3'], help="Whisper model type. Should be one of: ['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3'] (default: 'tiny')")
    parser.add_argument('--batch-size', '-bs', type=int, default=32, help='No of examples in a batch(default: 32)')
    parser.add_argument('--warmup-steps', type=int, default=10, help='TBD (default: 10)')
    parser.add_argument('--epoch-size', type=int, default=50000, help='TBD(default: 50000)')
    parser.add_argument('--num-epochs', type=int, default=35, help='TBD(default: 35)')  
    parser.add_argument('--max-iters', type=int, default=62500, help='TBD(default: 62500)')
    parser.add_argument('--max-steps', type=int, default=7812, help='TBD(default: 7812)')
    parser.add_argument('--micro-batch-size', '-mbs', type=int, default=4, help='TBD(default: 4)')
    parser.add_argument('--gradient-accumulation-iters', type=int, default=8, help='TBD(default: 8)')
    parser.add_argument('--max-seq-length', type=float, default=2048, help='TBD(default: 2048)')
    parser.add_argument('--max-input-length', type=float, default=1000, help='TBD(default: 1000)')
    parser.add_argument('--eval-interval', type=int, default=600, help='TBD(default: 600)') 
    parser.add_argument('--save-interval', type=int, default=1000, help='TBD(default: 1000)') 
    parser.add_argument('--eval-iters', type=int, default=100, help='TBD(default: 100)') 
    parser.add_argument('--eval-max-new-tokens', type=int, default=100, help='TBD(default: 100)') 
    parser.add_argument('--log-interval', type=int, default=1, help='TBD(default: 100)') 
    parser.add_argument('--precision', type=str, default=None, help='TBD(default: None)') 
    parser.add_argument('--quantize', type=str, default=None, choices=["bnb.nf4", "bnb.nf4-dq", "bnb.fp4", "bnb.fp4-dq", "bnb.int8-training"], help='TBD(default: None)') 
    parser.add_argument("--lazy_dump","-lz",type=bool,default=False, help="Lazy dump set to True create whisper features during batch creation.")    
    return parser

def setup():
    # Parse arguements
    parser = get_parser()
    args = parser.parse_args()
    
    # Load datasets
    train_data = torch.load(args.train_data_path, map_location=torch.device('cpu'))
    val_data = torch.load(args.val_data_path, map_location=torch.device('cpu'))
    train_data_len = len(train_data)
    val_data_len = len(val_data)
    
    # Update hyperparameters
    args.batch_size = args.batch_size // args.devices
    args.gradient_accumulation_iters = args.batch_size // args.micro_batch_size
    assert args.gradient_accumulation_iters > 0 , "Gradient accumulation iters is < 0"

    args.epoch_size = train_data_len // args.micro_batch_size // args.devices
    args.max_iters = args.num_epochs * args.epoch_size 
    args.max_steps = args.num_epochs * args.epoch_size * args.micro_batch_size // args.batch_size
    args.eval_iters = val_data_len // args.micro_batch_size // args.devices
    
    args.warmup_steps = 2 * (args.epoch_size // args.devices // args.batch_size)
    
    args.save_interval = args.epoch_size
    
    check_valid_checkpoint_dir(args.model_save_dir)
    
    # Setup fabric 
    plugins = None
    if args.quantize is not None and args.quantize.startswith("bnb."):
        if "mixed" in precision:
            raise ValueError("Quantization and mixed precision is not supported.")
        dtype = {"16-true": torch.float16, "bf16-true": torch.bfloat16, "32-true": torch.float32}[precision]
        plugins = BitsandbytesPrecision(args.quantize[4:], dtype)
        precision = None
        
    if args.devices > 1:
        if args.quantize:
            raise NotImplementedError(
                "Quantization is currently not supported for multi-GPU training. Please set devices=1 when using the"
                " --quantize flag."
            )
        strategy = FSDPStrategy(
            auto_wrap_policy={Block},
            activation_checkpointing_policy={Block},
            state_dict_type="full",
            limit_all_gathers=True,
            cpu_offload=False,
        )
    else:
        strategy = "auto"
    
    logger = CSVLogger(args.exp_dir.parent, args.exp_dir.name, flush_logs_every_n_steps=args.log_interval)
    fabric = L.Fabric(devices=args.devices, strategy=strategy, precision=args.precision, loggers=logger, plugins=plugins)
    fabric.launch()
    fabric.seed_everything(1337)  # same seed for every process to init model (FSDP)
    
    if fabric.global_rank == 0:
        os.makedirs(args.exp_dir, exist_ok=True)

    fabric.print(args)
    
    return args, fabric, train_data, val_data

def main():
    ##############################    Setup Training    ################################
    args, fabric, train_data, val_data = setup()
    
    config = Config.from_name(name=args.model_save_dir.name)
    config.downsampling_factor = args.downsampling_factor
    config.normalize_before = args.normalize_before 
    
    # Load whipser model to get count of decoder layers and its dimension
    whisper_model = whisper.load_model(args.whisper_model)
    whisper_model.eval()
    config.whisper_dim = whisper_model.decoder.ln.normalized_shape[0]
    config.no_whisper_decoder_layers = len(whisper_model.decoder.blocks)
    if not args.lazy_dump:
        del whisper_model

    checkpoint_path = args.model_save_dir / "lit_model.pth"
    fabric.print(f"Loading model {str(checkpoint_path)!r} with {config.__dict__}")
    with fabric.init_module(empty_init=(args.devices > 1)):
        model = GPT(config)
    mark_only_adapter_as_trainable(model)
    
    fabric.print(f"Number of trainable parameters: {num_parameters(model, requires_grad=True):,}")
    fabric.print(f"Number of non trainable parameters: {num_parameters(model, requires_grad=False):,}")

    model = fabric.setup_module(model)
    
    llama_tokenizer = Tokenizer(args.model_save_dir)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    if isinstance(fabric.strategy.precision, BitsandbytesPrecision):
        import bitsandbytes as bnb

        optimizer = bnb.optim.PagedAdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    optimizer = fabric.setup_optimizers(optimizer)
    scheduler = get_lr_scheduler(optimizer, warmup_steps=args.warmup_steps, max_steps=args.max_steps)

    # strict=False because missing keys due to Adapter weights not contained in state dict
    load_checkpoint(fabric, model, checkpoint_path, strict=False)

    fabric.seed_everything(1337 + fabric.global_rank)

    train_time = time.perf_counter()
    
    if args.lazy_dump:
        train(fabric, whisper_model, model, llama_tokenizer, optimizer, scheduler, train_data, val_data, args)
    else:
        train(fabric, None, model, llama_tokenizer, optimizer, scheduler, train_data, val_data, args)
    
    fabric.print(f"Training time: {(time.perf_counter()-train_time):.2f}s")
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")

    # Save the final checkpoint at the end of training
    save_path = args.exp_dir / "lit_model_adapter_finetuned.pth"
    save_adapter_checkpoint(fabric, model, save_path)

def train(
    fabric: L.Fabric,
    whisper_model: whisper,
    model: GPT,
    tokenizer: Tokenizer,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler,
    train_data: List[Dict],
    val_data: List[Dict],
    args: Namespace
) -> None:
    longest_seq_length, longest_seq_ix = max(get_longest_seq_length(train_data), get_longest_seq_length(val_data))
    model.max_seq_length = min(longest_seq_length, args.max_seq_length or float("inf")) # 4096 -> 2048 
    fabric.print(
        f"The longest sequence length in the train data is {longest_seq_length}, the model's maximum sequence length is"
        f" {model.max_seq_length} and context length is {model.config.block_size}"
    )

    validate(fabric, whisper_model, model, val_data, tokenizer, args, max_iters=2)  # sanity check

    throughput = ThroughputMonitor(fabric, window_size=50)
    step_count = 0
    total_lengths = 0
    total_t0 = time.perf_counter()
    
    with tqdm(range(1, args.max_iters+1), unit='iter') as titer:
        for iter_num in titer:
            input_ids, targets, _, whisper_decoder_states, whisper_decoder_state_mappings = get_batch(fabric, whisper_model, train_data, tokenizer, args, model.dtype, False, True, longest_seq_ix if iter_num == 1 else None)

            is_accumulating = iter_num % args.gradient_accumulation_iters != 0
            with fabric.no_backward_sync(model, enabled=is_accumulating):
                logits = model(input_ids, whisper_decoder_states, whisper_decoder_state_mappings, lm_head_chunk_size=128)
                # shift the targets such that output n predicts token n+1
                logits[-1] = logits[-1][..., :-1, :]
                loss = chunked_cross_entropy(logits, targets[..., 1:])
                fabric.backward(loss / args.gradient_accumulation_iters)

            if not is_accumulating:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                step_count += 1

            total_lengths += input_ids.numel()
            if iter_num % args.log_interval == 0:
                loss_item = loss.item()  # expensive device-to-host synchronization
                t1 = time.perf_counter()
                throughput.update(
                    time=t1 - total_t0, batches=iter_num, samples=iter_num * args.micro_batch_size, lengths=total_lengths
                )
                throughput.compute_and_log(step=iter_num)
                # fabric.print(
                #     f"iter {iter_num} step {step_count}: loss {loss_item:.4f}, iter time:"
                #     f" {(t1 - iter_t0) * 1000:.2f}ms{' (optimizer.step)' if not is_accumulating else ''}"
                # )
                titer.set_postfix(optimizer_steps=step_count, loss=loss_item)

            if not is_accumulating and step_count % args.eval_interval == 0:
                t0 = time.perf_counter()
                val_loss = validate(fabric, whisper_model, model, val_data, tokenizer, args, max_iters=args.eval_iters)
                t1 = time.perf_counter() - t0
                fabric.print(f"step {iter_num}: val loss {val_loss:.4f}, val time: {t1 * 1000:.2f}ms")
                fabric.barrier()
            if not is_accumulating and step_count % args.save_interval == 0:
                checkpoint_path = args.exp_dir / f"iter-{iter_num:06d}-ckpt.pth"
                save_adapter_checkpoint(fabric, model, checkpoint_path)


# the adapter "kv cache" cannot be initialized under `inference_mode`
@torch.no_grad()
def validate(
    fabric: L.Fabric, 
    whisper_model: whisper,
    model: GPT, 
    val_data: List[Dict], 
    tokenizer: Tokenizer,
    args: Namespace,
    max_iters: int
) -> torch.Tensor:
    fabric.print("Validating ...")
    model.eval()
    losses = torch.zeros(max_iters)
    for k in range(max_iters):
        input_ids, targets, _, whisper_decoder_states, whisper_decoder_state_mappings = get_batch(fabric, whisper_model, val_data, tokenizer, args, model.dtype, False, True)
        logits = model(input_ids, whisper_decoder_states, whisper_decoder_state_mappings)
        losses[k] = chunked_cross_entropy(logits[..., :-1, :], targets[..., 1:], chunk_size=0)
    val_loss = losses.mean()

    model.train()
    return val_loss.item()

def get_batch(
    fabric: L.Fabric, 
    whisper_model: whisper,
    data: List[Dict], 
    tokenizer: Tokenizer,
    args: Namespace,
    model_type,
    load_whisper_encoder_states: bool = False,
    load_whisper_decoder_states: bool = False,
    longest_seq_ix: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(len(data), (args.micro_batch_size,))
    if longest_seq_ix is not None:
        # force the longest sample at the beginning so potential OOMs happen right away
        ix[0] = longest_seq_ix

    input_ids = [data[i]["input_ids"].type(torch.int64) for i in ix]
    labels = [data[i]["labels"].type(torch.int64) for i in ix]

    if args.lazy_dump:

        whisper_decoder_features = []
        decoder_state_mappings = [data[i]["state_mappings"].type(torch.int64) for i in ix]

        for i in ix:
            audio_path = data[i]["path"]

            audio_dataset = Dataset.from_dict({"audio": [audio_path]}).cast_column("audio", Audio())

            #get features from the audio path.
            audio_array = audio_dataset[0]['audio']['array'].astype(numpy.single)
            audio_array = whisper.pad_or_trim(audio_array)

            mel = whisper.log_mel_spectrogram(audio_array, whisper_model.dims.n_mels).to(whisper_model.device)
            mel = mel.unsqueeze(0)

            whisper_ground_truth_tokens = data[i]["whisper_ground_truth_tokens"]
            whisper_ground_truth_tokens = torch.tensor(whisper_ground_truth_tokens).unsqueeze(0).to(mel.device)
            
            intermediate_logits = []
            def get_intermediate_output(module, input, output):
                intermediate_logits.append(output[0])

            # Register hooks for all the encoder layers we intend to tap into.
            hooks = []
            for layer in whisper_model.decoder.blocks:
                hook = layer.register_forward_hook(get_intermediate_output)
                hooks.append(hook)

            with torch.no_grad():
                encoder_state  = whisper_model.encoder(mel)
                _ = whisper_model.logits(whisper_ground_truth_tokens, encoder_state)
                
            #encoder_state = encoder_state.squeeze(0).detach().cpu().float()
            decoder_states = torch.stack(intermediate_logits, dim=0).cpu().detach().float()

            # Remove all hooks
            for hook in hooks:
                hook.remove()
                
            # Flush out the old intermediate logits
            intermediate_logits = []
            whisper_decoder_features.append(decoder_states)
    else:
        if load_whisper_encoder_states:
            whisper_encoder_features = [data[i]["encoder_states"].type(model_type) for i in ix]
        if load_whisper_decoder_states:
            whisper_decoder_features = [data[i]["decoder_states"].type(model_type) for i in ix]
            decoder_state_mappings = [data[i]["state_mappings"].type(torch.int64) for i in ix]

    # this could be `longest_seq_length` to have a fixed size for all batches
    max_len = max(len(s) for s in input_ids)

    def pad_right(x, pad_id):
        # pad right based on the longest sequence
        n = max_len - len(x)
        return torch.cat((x, torch.full((n,), pad_id, dtype=x.dtype)))
    
    x, y, wef, wdf = None, None, None, None
    if load_whisper_decoder_states:
        decoder_states_lens = [w.size(1) for w in whisper_decoder_features]
        max_len_decoder  = max(decoder_states_lens)
        def pad_right_whisper_decoder(w, pad_id):
            # pad right based on the longest sequence
            L, N, D = w.shape
            n = max_len_decoder - N
            return torch.cat((w, torch.full((L,n,D), pad_id, dtype=w.dtype)), dim=1)
    else:
        decoder_states_lens = None
    
    x = torch.stack([pad_right(ii, pad_id=0) for ii in input_ids])
    y = torch.stack([pad_right(l, pad_id=-1) for l in labels])
    if load_whisper_encoder_states:
        wef = torch.stack([w for w in whisper_encoder_features])
    if load_whisper_decoder_states: 
        wdf = torch.stack([pad_right_whisper_decoder(w, pad_id=0) for w in whisper_decoder_features])
        # wdf_state_mappings = torch.tensor(decoder_states_lens, dtype=torch.int64)
    
    if load_whisper_decoder_states:
        B, L = y.shape
        wdf_state_mappings = torch.full((B,L), 0, dtype=y.dtype)
        
        for ind, state_mapping in enumerate(decoder_state_mappings):
            wdf_state_mappings[ind] = decoder_states_lens[ind] - 1
            ln = len(state_mapping)
            wdf_state_mappings[ind][:ln] = state_mapping

    # Truncate if needed
    if args.max_input_length:
        x = x[:, :args.max_input_length]
        y = y[:, :args.max_input_length]
        if load_whisper_encoder_states:
            wef = wef[:, :args.max_input_length]
        if load_whisper_decoder_states:
            wdf = wdf[:, :args.max_input_length]
            wdf_state_mappings = torch.clamp(wdf_state_mappings, max=args.max_input_length-1)
    
    if load_whisper_encoder_states:
        if load_whisper_decoder_states:
            # Load both whisper's encoder and decoder states
            if fabric.device.type == "cuda" and x.device.type == "cpu":
                x, y, wef, wdf, wdf_state_mappings = fabric.to_device((x.pin_memory(), y.pin_memory(), wef.pin_memory(), wdf.pin_memory(), wdf_state_mappings.pin_memory()))
            else:
                x, y, wef, wdf, wdf_state_mappings = fabric.to_device((x, y, wef, wdf, wdf_state_mappings))
        else:
            # Load only whisper's encoder states
            if fabric.device.type == "cuda" and x.device.type == "cpu":
                x, y, wef = fabric.to_device((x.pin_memory(), y.pin_memory(), wef.pin_memory()))
            else:
                x, y, wef = fabric.to_device((x, y, wef))
    elif load_whisper_decoder_states:
        # Load only whisper's decoder states
        if fabric.device.type == "cuda" and x.device.type == "cpu":
            x, y, wdf, wdf_state_mappings = fabric.to_device((x.pin_memory(), y.pin_memory(), wdf.pin_memory(), wdf_state_mappings.pin_memory()))
        else:
            x, y, wdf, wdf_state_mappings = fabric.to_device((x, y, wdf, wdf_state_mappings))
    else:
        # Dont load any whisper's states
        if fabric.device.type == "cuda" and x.device.type == "cpu":
            x, y = fabric.to_device((x.pin_memory(), y.pin_memory()))
        else:
            x, y = fabric.to_device((x, y))
    return x, y, wef, wdf, wdf_state_mappings

def get_lr_scheduler(
    optimizer, 
    warmup_steps: int, 
    max_steps: int
):
    # linear warmup followed by cosine annealing
    scheduler1 = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: step / warmup_steps)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(max_steps - warmup_steps))
    return torch.optim.lr_scheduler.SequentialLR(optimizer, [scheduler1, scheduler2], milestones=[warmup_steps])

def get_longest_seq_length(
    data: List[Dict]
) -> Tuple[int, int]:
    # find out the minimum max_seq_length required during fine-tuning (saves memory!)
    lengths = [len(d["input_ids"]) for d in data]
    longest_seq_length = max(lengths)
    longest_seq_ix = lengths.index(longest_seq_length)
    return longest_seq_length, longest_seq_ix

def save_adapter_checkpoint(
    fabric: L.Fabric, 
    model: torch.nn.Module, 
    file_path: Path
) -> None:
    fabric.print(f"Saving adapter weights to {str(file_path)!r}")
    fabric.save(file_path, {"model": model}, filter={"model": adapter_filter}) # TODO: Update adapter_filter


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
