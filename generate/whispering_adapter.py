# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import sys
from pathlib import Path
from typing import Any, Optional

import lightning as L
import torch
import torch._dynamo.config
import torch._inductor.config

import whisper_openAI.whisper as Whisper

from whisper_openAI.whisper.decoding import Inference
from whisper_openAI.whisper.tokenizer import Tokenizer as Whisper_Tokenizer
from lit_gpt.tokenizer import Tokenizer as LLama_Tokenizer

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt import GPT

class PyTorchInference(Inference):
    def __init__(self, model: "Whisper", initial_token_length: int):
        self.model: "Whisper" = model
        self.initial_token_length = initial_token_length
        self.kv_cache = {}
        self.hooks = []
        self.aux_hooks = []
        self.intermediate_logits = []
        def get_intermediate_output(module, input, output):
            self.intermediate_logits.append(output[0])
            
        for layer in model.decoder.blocks:
            hook = layer.register_forward_hook(get_intermediate_output)
            self.aux_hooks.append(hook)
        

    def logits(self, tokens: torch.Tensor, audio_features: torch.Tensor) -> torch.Tensor:
        self.intermediate_logits = []
        if not self.kv_cache:
            self.kv_cache, self.hooks = self.model.install_kv_cache_hooks()

        if tokens.shape[-1] > self.initial_token_length:
            # only need to use the last token except in the first forward pass
            tokens = tokens[:, -1:]

        logits = self.model.decoder(tokens, audio_features, kv_cache=self.kv_cache)
        
        return torch.stack(self.intermediate_logits, dim=0), logits

    def cleanup_caching(self):
        for hook in self.hooks:
            hook.remove()
        
        for hook in self.aux_hooks:
            hook.remove()

        self.kv_cache = {}
        self.hooks = []

    def rearrange_kv_cache(self, source_indices):
        for module, tensor in self.kv_cache.items():
            # update the key/value cache to contain the selected sequences
            self.kv_cache[module] = tensor[source_indices].detach()

def _get_initial_whisper_tokens(tokenizer, n_ctx, sample_len, prefix=None, prompt=None):
    tokens = list(tokenizer.sot_sequence) + [tokenizer.no_timestamps]

    if prefix is not None:
        prefix_tokens = (
            tokenizer.encode(" " + prefix.strip())
            if isinstance(prefix, str)
            else prefix
        )
        if sample_len is not None:
            max_prefix_len = n_ctx // 2 - sample_len
            prefix_tokens = prefix_tokens[-max_prefix_len:]
        tokens = tokens + prefix_tokens

    if prompt is not None:
        prompt_tokens = (
            tokenizer.encode(" " + prompt.strip())
            if isinstance(prompt, str)
            else prompt
        )
        tokens = (
            [tokenizer.sot_prev]
            + prompt_tokens[-(n_ctx // 2 - 1) :]
            + tokens
        )

    return tuple(tokens)

def _init_whisper_model(model, tokenizer):
    n_ctx = model.dims.n_text_ctx
    sample_len = model.dims.n_text_ctx // 2

    initial_tokens = _get_initial_whisper_tokens(tokenizer, n_ctx, sample_len)

    # inference: implements the forward pass through the decoder, including kv caching
    return PyTorchInference(model, len(initial_tokens)), torch.tensor(initial_tokens, dtype=torch.int, device=model.device).unsqueeze(0)

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits

def multinomial_num_samples(probs: torch.Tensor) -> torch.Tensor:
    if torch._dynamo.is_compiling():
        # Faster alternative to `torch.multinomial(probs, num_samples=1)` that is also CUDAGraph friendly
        distribution = torch.empty_like(probs).exponential_(1)
        return torch.argmax(probs / distribution, dim=-1, keepdim=True)
    return torch.multinomial(probs, num_samples=1)

def sample(logits: torch.Tensor, top_k: Optional[int] = None, top_p: float = 0.9) -> torch.Tensor:
    logits = logits[0, -1]
    logits = top_k_top_p_filtering(logits, top_k, top_p)
    
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return multinomial_num_samples(probs)

def next_token(model: GPT, whisper_states: torch.Tensor, x: torch.Tensor, input_pos: torch.Tensor, top_k: float, top_p: float, **kwargs: Any) -> torch.Tensor:
    logits = model(x, whisper_states, None, input_pos)

    if top_k is None and top_p is None:
        next = torch.argmax(logits[0,-1], dim=-1, keepdim=True)
    else:
        next = sample(logits, top_k=top_k, top_p=top_p)
    return next.to(dtype=x.dtype)

# llama_model, llama_tokenizer, whisper_model, whisper_tokenizer, encoder_states,
@torch.inference_mode()
def generate(
    llama_model: GPT,
    llama_tokenizer: LLama_Tokenizer,
    whisper_model: Whisper,
    whisper_tokenizer: Whisper_Tokenizer,
    whisper_encoder_states: torch.Tensor, 
    max_returned_tokens: int,
    *,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    eos_id: Optional[int] = None,
) -> torch.Tensor:
    """Takes a conditioning sequence (prompt) and audio features as input and generates the ASR transcription.

    The implementation of this function is modified from A. Karpathy's nanoGPT.

    Args:
        model: The model to use.
        prompt: Tensor of shape (T) with indices of the prompt sequence.
        max_returned_tokens: The maximum number of tokens to return (given plus generated).
        temperature: Scales the predicted logits by 1 / temperature.
        top_k: If specified, only sample among the tokens with the k highest probabilities.
        eos_id: If specified, stop generating any more token once the <eos> token is triggered.
    """
    if llama_model.max_seq_length < max_returned_tokens - 1:
        # rolling the kv cache based on the `input_pos` value would be necessary. However, doing so would introduce a
        # data dependency on the `input_pos` tensor and impact model compilation. Since this setting is uncommon, we do
        # not support it to avoid negatively impacting the overall speed
        raise NotImplementedError(f"max_seq_length {llama_model.max_seq_length} needs to be >= {max_returned_tokens - 1}")

    device = whisper_encoder_states.device

    # Initialize whisper and get the decoder state after passing the start tokens
    whisper_encoder_states = whisper_encoder_states.unsqueeze(0)
    w_inference, w_tokens = _init_whisper_model(whisper_model, whisper_tokenizer)
    w_decoder_states, w_logits = w_inference.logits(w_tokens, whisper_encoder_states)
    w_decoder_states = w_decoder_states.unsqueeze(0)

    sentence_end_detected = False
    whisper_max_seq_reached = False

    # Initialize llama and get the first 
    token = torch.tensor([llama_tokenizer.bos_id],dtype=torch.int, device=device)
    tokens = [token]
    input_pos = torch.tensor([0], device=device)
    prev_generated_str = ''
    prev_generated_str_len = 0
    for _ in range(max_returned_tokens):
        token = next_token(llama_model, w_decoder_states, token.view(1, -1), input_pos, temperature=temperature, top_k=top_k, top_p=top_p).clone()
        tokens.append(token)
        generated_str = llama_tokenizer.decode(torch.cat(tokens))
        if (len(generated_str) > 0  and generated_str[-1] != '�') and generated_str != prev_generated_str:

            #string advanced by the current llama prediction
            advanced_string = generated_str[prev_generated_str_len:] 
            
            # advance whisper
            new_whisper_tokens = whisper_tokenizer.encode(advanced_string)
            for new_whisper_token in new_whisper_tokens:
                w_tokens = torch.cat([w_tokens, torch.tensor([new_whisper_token], dtype=w_tokens.dtype, device=device).unsqueeze(0)], dim=-1)
                if w_tokens.shape[1] > 448:
                    whisper_max_seq_reached = True
                    break
                w_decoder_states, w_logits = w_inference.logits(w_tokens, whisper_encoder_states)
                w_decoder_states = w_decoder_states.unsqueeze(0)    
            
            prev_generated_str_len = len(generated_str)   

        if whisper_max_seq_reached:
            break
        
        if token == eos_id:
            sentence_end_detected = True
            break
        
        input_pos = input_pos.add_(1)
        prev_generated_str = generated_str
    w_inference.cleanup_caching()
    
    return torch.cat(tokens), sentence_end_detected
    
    # T = prompt.size(0)
    # assert max_returned_tokens > T


    # whisper_states = whisper_states.unsqueeze(0)
    # device = prompt.device
    # tokens = [prompt]
    # input_pos = torch.tensor([T], device=device)
    # token = next_token(
    #     model, whisper_states, prompt.view(1, -1), torch.arange(0, T, device=device), temperature=temperature, top_k=top_k
    # ).clone()
    # tokens.append(token)
    # for _ in range(2, max_returned_tokens - T + 1):
    #     '''
    #         The input to the model is only generated tokens, as the combination of kv_cache and input_pos 
    #         is used to handle the previous context during inference.
    #     '''
    #     token = next_token(model, whisper_states, token.view(1, -1), input_pos, temperature=temperature, top_k=top_k).clone()
    #     tokens.append(token)
    #     if token == eos_id:
    #         break
    #     input_pos = input_pos.add_(1)
    # return torch.cat(tokens)

if __name__ == "__main__":
    raise NotImplementedError('This file is just a placeholder for inference helper functions. The complete inference code is in inference/')
