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
from MagicDec.Engine.SqueezedAttention.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
import torch.distributed as dist
import torch.multiprocessing as mp

class LMBackend_Squeeze:
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

    # def load_model(self, checkpoints: str, use_tp: bool, rank_group=None, group = None):
    #     self.model: Transformer = load_model_snapKV(checkpoint_path=checkpoints, device=self.device, precision=self.dtype, use_tp=use_tp, rank_group=rank_group, group=group)        
    def set_attrs(self, obj, attrs: dict):
      for name, val in attrs.items():
          setattr(obj, name, val)

    def load_model(self, model_path: str, path_to_clusters: str, percentile):
        # config params
        config_params = {}
        config_params['path_to_clusters_cosine'] = path_to_clusters
        config_params['use_centroids'] = False
        config_params['hierarchical_lookup'] = False
        config_params['percent_clusters'] = 5 #not used actually in target model
        config_params['percent_clusters_l2'] = 5 #not used actually in target model
        config_params['percentile'] = percentile #not used actually in target model
        config_params['percentile_lower'] = 0.7 #not used actually in target model
        config_params['obs_window'] = 100 #not used actually in target model

        if "LLaMA-2-7B-32K" in model_path or "LWM" in model_path or "longchat" in model_path:
            config = LlamaConfig.from_pretrained(model_path)

            # set attn implementation
            config._flash_attn_2_enabled = True
            config._attn_implementation = "flash_attention_2"
            dtype = torch.bfloat16

            # clustering config parameters
            config.path_to_clusters_cosine = config_params['path_to_clusters_cosine']
            config.use_centroids = config_params['use_centroids']
            config.hierarchical_lookup = config_params['hierarchical_lookup']
            config.percent_clusters = config_params['percent_clusters']
            config.percent_clusters_l2 = config_params['percent_clusters_l2']
            config.percentile = config_params['percentile']
            config.percentile_lower = config_params['percentile_lower']
            config.obs_window = config_params['obs_window']

            # load model
            model = LlamaForCausalLM.from_pretrained(model_path, config=config, torch_dtype=dtype)
            model = model.to(device)
            tokenizer = LlamaTokenizer.from_pretrained(model_path)

        else:
            assert (False) # not implemented yet for other models

        self.model = model.eval()
        self.target_cfg = config_params

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
            # outputs_2 = self.model.generate(input_from_prefill,max_new_tokens=1,num_beams=1,do_sample=False,temperature=1.0,use_cache=True)[0]
            
            verified_length = input_ids.shape[-1]
            verified_logits = outputs.logits[:,-verified_length:]
            
            verified_tokens = torch.argmax(verified_logits, dim=-1)
            # verified_tokens_2 = outputs_2[-verified_length:].unsqueeze(0)

            return verified_tokens

    @torch.inference_mode()
    def speculate(self, input_ids: torch.LongTensor, bsz, gamma):
      tokens_buffer= torch.zeros((bsz, gamma), device="cuda").long()
      # draft_past_key_values = self.draft_past_key_values
      next_input_token = input_ids

      self.model.model.use_centroids = True
      for i in range(self.model.model.config.num_hidden_layers):
        self.model.model.layers[i].self_attn.use_centroids = True      
      for i in range(gamma):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
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
    
        
    # def compile(self):
    #     import torch._dynamo.config
    #     import torch._inductor.config
    #     torch._inductor.config.coordinate_descent_tuning = True
    #     torch._inductor.config.triton.unique_kernel_names = True
    #     torch._inductor.config.fx_graph_cache = True
    #     torch._functorch.config.enable_autograd_cache = True
    #     self.model_forward = torch.compile(self.model_forward, mode="max-autotune", fullgraph=True)
    #     if self.is_spec:
    #         self.draft_forward = torch.compile(self.draft_forward, mode="max-autotune", fullgraph=True)

    # # Only used for baseline inference
    # @torch.inference_mode()
    # def inference(self, input_ids: torch.LongTensor, benchmark = False):
    #         dec_len = input_ids.shape[1]
    #         self.pre_decode(dec_len=dec_len)

    #         logits = self.model_forward(
    #             model=self.model, 
    #             x=input_ids,
    #             input_pos=self.cachelens, 
    #             kv_append_indptr = self.qo_indptr*dec_len, kv_page_indices = self.paged_kv_indices, kv_page_indptr= self.paged_kv_indptr, kv_page_lastlen = self.paged_kv_last_page_len)
            
    #         self.cachelens += dec_len
    #         if benchmark:
    #             # If benchmarking the latency, don't update the cachelens and page table
    #             self.cachelens -= dec_len
    #             self.paged_kv_last_page_len -= dec_len
    #         return logits
    
    # def pre_decode(self, dec_len):
    #         self.paged_kv_last_page_len += dec_len
    #         self.decode_wrapper.plan(
    #             qo_indptr=self.qo_indptr*dec_len,
    #             paged_kv_indptr=self.paged_kv_indptr,
    #             paged_kv_indices=self.paged_kv_indices,
    #             paged_kv_last_page_len=self.paged_kv_last_page_len,
    #             num_qo_heads=self.model.config.n_head, 
    #             num_kv_heads=self.model.config.n_local_heads, 
    #             head_dim=self.model.config.head_dim, 
    #             page_size=self.page_size, 
    #             q_data_type=self.dtype, 
    #             causal=True,
    #         )
    
    # def pre_verify(self, dec_len):
    #         self.paged_kv_last_page_len += dec_len
    #         self.draft_paged_kv_last_page_len += 1
    #         self.draft_cachelens += 1

    #         self.decode_wrapper.plan(
    #             qo_indptr=self.qo_indptr*dec_len,
    #             paged_kv_indptr=self.paged_kv_indptr,
    #             paged_kv_indices=self.paged_kv_indices,
    #             paged_kv_last_page_len=self.paged_kv_last_page_len,
    #             num_qo_heads=self.model.config.n_head, 
    #             num_kv_heads=self.model.config.n_local_heads, 
    #             head_dim=self.model.config.head_dim, 
    #             page_size=self.page_size, 
    #             q_data_type=self.dtype, 
    #             causal=True,
    #         )
    
    # def pre_spec(self, dec_len):
    #         self.draft_paged_kv_last_page_len += dec_len
    #         self.draft_wrapper.plan(
    #             qo_indptr=self.qo_indptr*dec_len,
    #             paged_kv_indptr=self.draft_paged_kv_indptr,
    #             paged_kv_indices=self.draft_paged_kv_indices,
    #             paged_kv_last_page_len=self.draft_paged_kv_last_page_len,
    #             num_qo_heads=self.model.config.n_head, 
    #             num_kv_heads=self.model.config.n_local_heads, 
    #             head_dim=self.model.config.head_dim, 
    #             page_size=self.page_size, 
    #             q_data_type=self.dtype, 
    #             causal=True,
    #         )
        
    
    # def pre_encode(self, dec_len):
    #     self.num_pages_per_request+=1
    #     qo_indptr = self.qo_indptr*dec_len
    #     self.paged_kv_indices = torch.cat([torch.arange(i * self.max_num_pages_per_request, i * self.max_num_pages_per_request + self.num_pages_per_request[i], dtype=torch.int32, device=self.device) for i in range(self.batch_size)])
    #     self.paged_kv_indptr[1:] = torch.cumsum(self.num_pages_per_request, dim=0, dtype=torch.int32)
    #     self.paged_kv_last_page_len = torch.full((self.batch_size,), dec_len, dtype=torch.int32, device=self.device)
    #     self.prefill_wrapper.plan(
    #         qo_indptr=qo_indptr,
    #         paged_kv_indptr=self.paged_kv_indptr,
    #         paged_kv_indices=self.paged_kv_indices,
    #         paged_kv_last_page_len=self.paged_kv_last_page_len,
    #         num_qo_heads=self.model.config.n_head, 
    #         num_kv_heads=self.model.config.n_local_heads, 
    #         head_dim=self.model.config.head_dim, 
    #         page_size=self.page_size, 
    #         q_data_type=self.dtype, 
    #         causal=True
    #         )
          
    
    # @torch.inference_mode()
    # def clear_kv(self):
    #     for b in self.model.layers:
    #         b.attention.kv_cache.kv_cache.zero_()
    #         if self.is_spec:
    #             b.attention.kv_cache.draft_cache.zero_()
    #     self.cachelens.zero_()
    #     self.qo_indptr = torch.arange(self.batch_size+1, dtype=torch.int32, device=self.device)
    #     self.paged_kv_indptr = torch.arange(self.batch_size+1, dtype=torch.int32, device=self.device)
    #     self.paged_kv_indices = torch.empty(self.max_num_pages, dtype=torch.int32, device=self.device)
    #     self.paged_kv_last_page_len = torch.zeros((self.batch_size), dtype=torch.int32, device=self.device)
    #     self.num_pages_per_request = torch.zeros(self.batch_size, device=self.device, dtype=torch.int32)
    #     if self.is_spec:
    #         self.draft_cachelens.zero_()
    #         self.draft_paged_kv_indptr = torch.arange(self.batch_size+1, dtype=torch.int32, device=self.device)*(self.draft_budget//self.page_size + 1)
    #         self.draft_paged_kv_indices = torch.arange(self.draft_num_pages, dtype=torch.int32, device=self.device)
    #         self.draft_paged_kv_last_page_len = torch.ones((self.batch_size), dtype=torch.int32, device=self.device)

    
    # @torch.inference_mode()
    # def setup_caches(self, max_batch_size: int = 1, max_seq_length: int = 2048, draft_budget = 0, window_size = 32):
    #     self.max_length = max_seq_length
    #     self.batch_size = max_batch_size
    #     self.cachelens = torch.zeros(max_batch_size, dtype=torch.int32, device=self.device)
    #     # Prefill length should be devisible by 128 and plus 1 or window_size
    #     # Max Length should be divisible by 128
    #     page_size = 128
    #     max_num_pages = max_batch_size * max_seq_length // page_size
    #     if max_num_pages*page_size < max_batch_size*max_seq_length:
    #         max_num_pages += max_batch_size
    #     self.max_num_pages_per_request = max_num_pages // max_batch_size
    #     self.num_pages_per_request = torch.zeros(max_batch_size, device=self.device, dtype=torch.int32)
    #     self.page_size = 128
    #     self.max_num_pages = max_num_pages


    #     # Init Target Attention Backend(Flashinfer)
    #     self.decode_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=self.device)
    #     self.prefill_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=self.device)

    #     self.qo_indptr = torch.arange(max_batch_size+1, dtype=torch.int32, device=self.device)
    #     self.paged_kv_indptr = torch.arange(max_batch_size+1, dtype=torch.int32, device=self.device)
    #     self.paged_kv_indices = torch.empty(max_num_pages, dtype=torch.int32, device=self.device)
    #     self.paged_kv_last_page_len = torch.zeros((max_batch_size), dtype=torch.int32, device=self.device)
    #     self.decode_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(self.decode_buffer, "NHD", use_cuda_graph=True,
    #                                                                           qo_indptr_buf=self.qo_indptr, 
    #                                                                           paged_kv_indptr_buf=self.paged_kv_indptr, 
    #                                                                           paged_kv_indices_buf=self.paged_kv_indices, 
    #                                                                           paged_kv_last_page_len_buf=self.paged_kv_last_page_len)
        
    #     self.prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(self.prefill_buffer, "NHD")
    #     torch.library.define(
    #         "mylib::target_decode",
    #         "(Tensor q, Tensor kv_cache) -> Tensor",
    #     )
    #     @torch.library.impl("mylib::target_decode", "cuda")
    #     def target_decode(q, kv_cache):
    #         return self.decode_wrapper.run(
    #             q, kv_cache
    #         )
    #     @torch.library.register_fake("mylib::target_decode")
    #     def target_decode_abstract(q, kv_cache):
    #         return torch.empty_like(q)
        
    #     torch.library.define(
    #         "mylib::target_prefill",
    #         "(Tensor q, Tensor kv_cache) -> Tensor",
    #     )
    #     @torch.library.impl("mylib::target_prefill", "cuda")
    #     def target_prefill(q, kv_cache):
    #         return self.prefill_wrapper.run(
    #             q, kv_cache
    #         )
    #     @torch.library.register_fake("mylib::target_prefill")
    #     def target_prefill_abstract(q, kv_cache):
    #         return torch.empty_like(q)

    #     # If using speculative decoding, init draft attention backend
    #     if self.is_spec:
    #         self.draft_budget = draft_budget
    #         self.draft_cachelens = torch.zeros(max_batch_size, dtype=torch.int32, device=self.device)
    #         self.draft_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=self.device)
    #         self.draft_num_pages = (draft_budget//page_size + 1)*max_batch_size
    #         self.draft_paged_kv_indptr = torch.arange(max_batch_size+1, dtype=torch.int32, device=self.device)*(draft_budget//page_size + 1)
    #         self.draft_paged_kv_indices = torch.arange(self.draft_num_pages, dtype=torch.int32, device=self.device)
    #         self.draft_paged_kv_last_page_len = torch.ones((max_batch_size), dtype=torch.int32, device=self.device)
    #         self.draft_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(self.draft_buffer, "NHD", use_cuda_graph=True,
    #                                                                             qo_indptr_buf=self.qo_indptr, 
    #                                                                             paged_kv_indptr_buf=self.draft_paged_kv_indptr, 
    #                                                                             paged_kv_indices_buf=self.draft_paged_kv_indices, 
    #                                                                             paged_kv_last_page_len_buf=self.draft_paged_kv_last_page_len)
    #         torch.library.define(
    #             "mylib::draft_decode",
    #             "(Tensor q, Tensor kv_cache) -> Tensor",
    #         )
    #         @torch.library.impl("mylib::draft_decode", "cuda")
    #         def draft_decode(q, kv_cache):
    #             return self.draft_wrapper.run(
    #                 q, kv_cache
    #             )
    #         @torch.library.register_fake("mylib::draft_decode")
    #         def draft_decode_abstract(q, kv_cache):
    #             return torch.empty_like(q)

    #     if self.is_spec:
    #         with torch.device(self.device):
    #             self.model.setup_caches(num_pages=max_num_pages, page_size=page_size, spec=self.is_spec, draft_num_pages = self.draft_num_pages, draft_budget = draft_budget, window_size = window_size)
    #     else:
    #         with torch.device(self.device):
    #             self.model.setup_caches(num_pages=max_num_pages, page_size=page_size)