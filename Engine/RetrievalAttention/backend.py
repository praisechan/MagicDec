from datasets import load_dataset
from torch.nn import CrossEntropyLoss

device = "cuda"

# from MagicDec.Engine.SnapKV.model import Transformer
# from MagicDec.Engine.utils import load_model_snapKV
# import flashinfer

import os
from datasets import load_dataset
import torch
import json
from transformers import (
    AutoTokenizer,
    AutoConfig,
    LlamaTokenizer,
    LlamaForCausalLM,
    LlamaConfig,
    AutoModelForCausalLM,
)
from tqdm import tqdm
import numpy as np
import random
import argparse

import copy
import torch.distributed as dist
import torch.multiprocessing as mp
from MagicDec.Engine.RetrievalAttention.model_hub import LlamaModel, QwenModel
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from MagicDec.Engine.RetrievalAttention.benchmark.config import generate_config, parse_attn_args

class LMBackend_Retro:
    def __init__(self, dtype = torch.bfloat16, device: str = "cuda:0", dec_len: int = 1, draft_dec_len: int = None) -> None:
        self.dtype = dtype
        self.device = device
        self.dec_len = dec_len
        self.model_forward = lambda model, x, input_pos, kv_append_indptr, kv_page_indices, kv_page_indptr, kv_page_lastlen: model(x, input_pos, kv_append_indptr, kv_page_indices, kv_page_indptr, kv_page_lastlen)
        self.prefill = lambda model, x, input_pos, kv_append_indptr, kv_page_indices, kv_page_indptr, kv_page_lastlen, is_last=None, draft_paged_kv_indptr=None, draft_paged_kv_indices=None, draft_paged_kv_last_page_len=None: model.prefill(x, input_pos, kv_append_indptr, kv_page_indices, kv_page_indptr, kv_page_lastlen, is_last, draft_paged_kv_indptr, draft_paged_kv_indices, draft_paged_kv_last_page_len)
        self.cachelens = None
        self.is_spec = False
        if draft_dec_len != None:
            self.is_spec = True
            self.draft_cachelens = None
            self.model_forward = lambda model, x, input_pos, kv_append_indptr, kv_page_indices, kv_page_indptr, kv_page_lastlen, draft_kv_page_indices, draft_kv_page_indptr, draft_kv_page_lastlen: model.verify(x, input_pos, kv_append_indptr, kv_page_indices, kv_page_indptr, kv_page_lastlen, draft_kv_page_indices, draft_kv_page_indptr, draft_kv_page_lastlen)
            self.draft_forward = lambda model, x, input_pos, kv_append_indptr, kv_page_indices, kv_page_indptr, kv_page_lastlen: model.draft_forward(x, input_pos, kv_append_indptr, kv_page_indices, kv_page_indptr, kv_page_lastlen)
        
        # for Quest
        self.draft_past_key_values = None
        self.input_tokens = None
        self.verified_cachelength = 0

    def load_model(self, model_path, max_len, dtype, device):
        if 'Llama' in model_path:
            llm = LlamaModel(model_path,
                max_length=max_len,
                dtype=dtype,
                device_map=device)
        elif 'Qwen' in model_path:
            llm = QwenModel(model_path,
                max_length=max_len,
                dtype=dtype,
                device_map=device)
        else:
            raise ValueError(f"Unsupported model: {model_path}")

        llm.tokenizer.pad_token = llm.tokenizer.eos_token
        llm.tokenizer.padding_side = "left"
        
        self.model = llm

    def load_draft_model(self, model_path: str, bsz, max_len):
        self.input_tokens = torch.zeros(bsz, max_len+1, device="cuda").long()
        self.cachelens = torch.zeros(bsz, dtype=torch.int32, device=self.device)
        if not hasattr(self, "model"):
            self.load_model(model_path)                

        # # 2) shallow‐copy your top‐level wrapper
        # self.draft_model = copy.copy(self.model)

        # # 3) shallow‐copy the inner LlamaModel so it can hold its own config
        # inner_copy = copy.copy(self.model.model)
        # inner_copy.config = draft_cfg            # only affects the draft copy

        # # 4) re‐wire the draft wrapper
        # self.draft_model.model  = inner_copy
        # self.draft_model.config = draft_cfg     # so generate() sees it too
        
        # self.draft_model.to(self.device).eval()

        # draft_model = LlamaForCausalLM(draft_cfg)
        # draft_model.model = self.model.model
        # draft_model.lm_head = self.model.lm_head
        # self.draft_model = draft_model

    def preprocess_input(self, json_obj, prompt_format, attn_type, model_path, budget_ratio, estimate_ratio):
        # different_prefix_index = json_obj.pop('different_prefix_index')
        # prompt = prompt_format.format(**json_obj)
        # prompt_noquery = prompt_only_format.format(**json_obj)

        # # perform truncation
        # prompt, truncated_shared_prefix_length = truncate_fn(prompt, prompt_noquery, tokenizer, max_length, task, DEVICE, MODEL)

        # # encode input
        # input = tokenizer(prompt, truncation=False, return_tensors="pt").to(DEVICE)
        
        # return input['input_ids'], truncated_shared_prefix_length, different_prefix_index

        prompt = prompt_format.format(**json_obj)

        inputs = self.model.tokenizer([prompt], return_tensors="pt", padding=True)
        input_ids = inputs.input_ids
        attention_masks = inputs.attention_mask

        attn_config = generate_config(
            model_path, 
            input_ids.shape[1], 
            attn_type,
            budget_ratio=budget_ratio,
            estimate_ratio=estimate_ratio,
        )
        
        return input_ids, attention_masks, attn_config

    # Only used for target verification
    @torch.inference_mode()
    def verify(self, input_ids: torch.LongTensor, benchmark = False):
            input_from_prefill = torch.concat((self.input_tokens[:, :self.verified_cachelength], input_ids), dim=1)
            dec_len = input_ids.shape[1]
            # self.pre_verify(dec_len=dec_len)
            # self.model.config=self.target_cfg
            # self.model.model.config=self.target_cfg
            # self.set_attrs(self.model, self.target_cfg)
            self.set_attrs(self.model.model, self.target_cfg)
            outputs = self.model(
                input_from_prefill,
                # past_key_values=self.draft_past_key_values,
                use_cache=True,
                # output_all_token=True
            )
            outputs_2 = self.model.generate(input_from_prefill,max_new_tokens=1,num_beams=1,do_sample=False,temperature=1.0,use_cache=True)[0]
            
            verified_length = input_ids.shape[-1]
            verified_logits = outputs.logits[:,-verified_length:]
            
            verified_tokens = torch.argmax(verified_logits, dim=-1)
            verified_tokens_2 = outputs_2[-verified_length:].unsqueeze(0)

            return verified_tokens_2

    @torch.inference_mode()
    def speculate(self, input_ids: torch.LongTensor, bsz, gamma):
      tokens_buffer= torch.zeros((bsz, gamma), device="cuda").long()
      # draft_past_key_values = self.draft_past_key_values
      next_input_token = input_ids

      self.model.model.use_centroids = True
      for i in range(self.model.model.config.num_hidden_layers):
        self.model.model.layers[i].self_attn.use_centroids = True      
      for i in range(gamma):
        outputs = self.model(
            next_input_token,
            # past_key_values=draft_past_key_values,
            use_cache=False,
        )
        
        last_token = torch.argmax(outputs.logits[:,-1,:], dim=-1)
        next_input_token = torch.concat((next_input_token, last_token.unsqueeze(-1)), dim=1)
        tokens_buffer[:, i:i+1] = last_token
        # draft_past_key_values = outputs.past_key_values

      self.model.model.use_centroids = False
      for i in range(self.model.model.config.num_hidden_layers):
        self.model.model.layers[i].self_attn.use_centroids = False

      return tokens_buffer
    
    @torch.inference_mode()
    def draft_kv_update(self, input_ids: torch.LongTensor):
        input_from_prefill = torch.concat((self.input_tokens[:, :self.verified_cachelength], input_ids), dim=1)
        # outputs = self.draft_model(
        #     input_from_prefill,
        #     past_key_values=None,
        #     use_cache=True,
        # )
        self.verified_cachelength += input_ids.shape[1]
        self.input_tokens[:,:self.verified_cachelength] = input_from_prefill
        # self.draft_past_key_values = outputs.past_key_values

    @torch.inference_mode()
    def encode(self, input_ids: torch.LongTensor, truncated_shared_prefix_length, different_prefix_index):        
        self.set_attrs(self.model.model, self.target_cfg)
        outputs = self.model(
            input_ids,
            past_key_values=None,
            # use_cache=True,
        )
        # self.draft_past_key_values = outputs.past_key_values
        self.input_tokens[:,:input_ids.shape[1]] = input_ids
        self.verified_cachelength = input_ids.shape[1]
        self.cachelens = input_ids.shape[1]

        # reset is needed for draft model
        # find more info in modeling_llama.py/reset_context
        self.model.model.use_centroids = True
        self.model.model.shared_prefix_length = truncated_shared_prefix_length
        self.model.model.different_prefix_index = different_prefix_index
        outputs = self.model(
            input_ids,
            past_key_values=None,
            # use_cache=True,
        )
        self.model.model.use_centroids = False
        
        return torch.argmax(outputs.logits, dim=-1)