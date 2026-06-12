import gc
import re
import torch
import torch.nn.functional as F
import flashinfer
from transformers import AutoTokenizer, Qwen3MoeForCausalLM, Qwen3MoeConfig
from .LLM import LLM
from cache_hub import flash_attn_cache, retroinfer_cache, retroinfer_cache_gpu
from attn_hub import full_decode_attn, retroinfer_decode_attn, \
                     full_prefill_attn, prefill_xattn, prefill_minfer


class Qwen3MoeLayer:
    """A class representing a single Qwen3-MoE transformer layer."""

    def __init__(self, layer_idx, device) -> None:
        self.layer_idx = layer_idx
        self.device = device

    def init_layer(self, hf_layer):
        # Attention weights (no bias in Qwen3-MoE)
        wq = hf_layer.self_attn.q_proj.weight.detach()
        wk = hf_layer.self_attn.k_proj.weight.detach()
        wv = hf_layer.self_attn.v_proj.weight.detach()
        self.wqkv = torch.cat((wq, wk, wv), dim=0).to(self.device, non_blocking=True)
        self.wo = hf_layer.self_attn.o_proj.weight.detach().to(self.device, non_blocking=True)

        # Per-head QK RMSNorm (new in Qwen3)
        self.q_norm_weight = hf_layer.self_attn.q_norm.weight.detach().to(self.device, non_blocking=True)
        self.k_norm_weight = hf_layer.self_attn.k_norm.weight.detach().to(self.device, non_blocking=True)

        # MoE router gate
        self.gate_weight = hf_layer.mlp.gate.weight.detach().to(self.device, non_blocking=True)

        # Stack expert weights: [num_experts, 2*moe_intermediate_size, hidden_size]
        # and [num_experts, hidden_size, moe_intermediate_size]
        expert_gate_up_list = []
        expert_down_list = []
        for expert in hf_layer.mlp.experts:
            gate_proj = expert.gate_proj.weight.detach()
            up_proj = expert.up_proj.weight.detach()
            expert_gate_up_list.append(torch.cat((gate_proj, up_proj), dim=0))
            expert_down_list.append(expert.down_proj.weight.detach())
        self.expert_gate_up_proj = torch.stack(expert_gate_up_list, dim=0).to(self.device, non_blocking=True)
        self.expert_down_proj = torch.stack(expert_down_list, dim=0).to(self.device, non_blocking=True)

        # Layer norms
        self.input_layernorm_weight = hf_layer.input_layernorm.weight.detach().to(self.device, non_blocking=True)
        self.input_layernorm_variance_epsilon = hf_layer.input_layernorm.variance_epsilon
        self.post_attention_layernorm_weight = hf_layer.post_attention_layernorm.weight.detach().to(self.device, non_blocking=True)
        self.post_attention_layernorm_variance_epsilon = hf_layer.post_attention_layernorm.variance_epsilon

        del wq, wk, wv, expert_gate_up_list, expert_down_list


class Qwen3MoeModel(LLM):
    """A class representing the Qwen3-MoE model (e.g. Qwen3-30B-A3B)."""

    def __init__(
        self,
        model_name: str,
        max_length: int,
        dtype: torch.dtype,
        device_map: str,
        tokenizer: AutoTokenizer = None
    ) -> None:
        super().__init__(model_name, max_length, dtype, device_map)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name) if tokenizer is None else tokenizer
        self.config = Qwen3MoeConfig.from_pretrained(model_name)
        self.num_layers = self.config.num_hidden_layers
        self.num_heads = self.config.num_attention_heads
        self.num_key_value_heads = self.config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.hidden_size = self.config.hidden_size
        # Qwen3-MoE has explicit head_dim (128) which differs from hidden_size // num_heads (64)
        self.head_dim = getattr(self.config, 'head_dim', self.hidden_size // self.num_heads)
        self.base = self.config.rope_theta
        self.max_position_embeddings = self.config.max_position_embeddings
        self.vocab_size = self.config.vocab_size
        self.norm_eps = self.config.rms_norm_eps
        self.eos_tokens = [self.config.eos_token_id]

        # MoE config
        self.num_experts = self.config.num_experts
        self.num_experts_per_tok = self.config.num_experts_per_tok
        self.norm_topk_prob = getattr(self.config, 'norm_topk_prob', True)

        self.init_model()


    def _set_cos_sin_cache(self):
        # Qwen3-MoE uses standard RoPE (no YaRN)
        t = torch.arange(self.max_length, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        return freqs.cos() * self.attention_scaling, freqs.sin() * self.attention_scaling


    def init_model(self):
        hf_model = Qwen3MoeForCausalLM.from_pretrained(self.model_name, torch_dtype=self.dtype)

        self.num_gpus = torch.cuda.device_count() if self.device_map == 'auto' else 1
        if self.device_map == 'auto' and self.num_gpus == 1:
            self.device_map = 'cuda:0'

        if self.device_map != "auto":   # single GPU
            self.layer_mapping = {}
            for ldx in range(0, self.num_layers):
                self.layer_mapping.update({str(ldx): self.device_map})

            self.embed_tokens = hf_model.model.embed_tokens.weight.detach().to(self.device_map, non_blocking=True)
            self.lm_head = hf_model.lm_head.weight.detach().to(self.device_map, non_blocking=True)

            self.norm_weight = hf_model.model.norm.weight.detach().to(self.device_map, non_blocking=True)
            self.norm_variance_epsilon = hf_model.model.norm.variance_epsilon

            self.position_ids = torch.arange(0, self.max_length).to(self.device_map, non_blocking=True)
            self.inv_freq = hf_model.model.rotary_emb.inv_freq.detach().to(self.device_map, non_blocking=True)
            self.attention_scaling = getattr(hf_model.model.rotary_emb, 'attention_scaling', 1.0)
            self.cos_cache, self.sin_cache = self._set_cos_sin_cache()
            self.cos_sin_cache = torch.cat((self.cos_cache, self.sin_cache), dim=-1)

            self.layers = []
            for idx, hf_layer in enumerate(hf_model.model.layers):
                layer = Qwen3MoeLayer(idx, device=self.device_map)
                layer.init_layer(hf_layer)
                self.layers.append(layer)
                hf_model.model.layers[idx] = None
                gc.collect()

        else:   # multi GPUs
            self.gpu_ids = list(range(self.num_gpus))
            self.layer_interval = (self.num_layers + self.num_gpus - 1) // self.num_gpus
            self.layer_mapping = {}
            for ldx in range(0, self.num_layers):
                self.layer_mapping.update({str(ldx): f'cuda:{ldx // self.layer_interval}'})

            self.embed_tokens = hf_model.model.embed_tokens.weight.detach().to(f'cuda:{self.gpu_ids[0]}', non_blocking=True)
            self.lm_head = hf_model.lm_head.weight.detach().to(f'cuda:{self.gpu_ids[0]}', non_blocking=True)

            self.norm_weight = hf_model.model.norm.weight.detach().to(f'cuda:{self.gpu_ids[0]}', non_blocking=True)
            self.norm_variance_epsilon = hf_model.model.norm.variance_epsilon

            self.position_ids = torch.arange(0, self.max_length).to(f'cuda:{self.gpu_ids[0]}', non_blocking=True)
            self.inv_freq = hf_model.model.rotary_emb.inv_freq.detach().to(f'cuda:{self.gpu_ids[0]}', non_blocking=True)
            self.attention_scaling = getattr(hf_model.model.rotary_emb, 'attention_scaling', 1.0)
            self.cos_cache, self.sin_cache = self._set_cos_sin_cache()
            self.cos_sin_cache = torch.cat((self.cos_cache, self.sin_cache), dim=-1)

            self.layers = []
            for ldx, hf_layer in enumerate(hf_model.model.layers):
                layer = Qwen3MoeLayer(ldx, device=self.layer_mapping[str(ldx)])
                layer.init_layer(hf_layer)
                self.layers.append(layer)
                hf_model.model.layers[ldx] = None
                gc.collect()

        del self.inv_freq, self.cos_cache, self.sin_cache
        del hf_model
        gc.collect()
        torch.cuda.empty_cache()

        # Default thresholds and patterns (no model-specific profiles yet)
        self.thresholds = [torch.ones((self.num_heads,), device=self.layer_mapping[str(layer_idx)]) * 0.9
                           for layer_idx in range(self.num_layers)]
        self.best_patterns = [{str(head_idx): ["vertical_and_slash", 1000, 6096, 1] for head_idx in range(self.num_heads)}
                              for layer_idx in range(self.num_layers)]


    def init_kv_cache(self, valid_start, attn_config):
        self.kv_cache = None
        gc.collect()

        config = attn_config

        if self.attention_type == 'Full_Flash_Attn':
            self.kv_cache = flash_attn_cache(
                valid_start=valid_start,
                layer_num=self.num_layers,
                batch_size=self.batch_size,
                max_length=self.max_new_length + self.input_length,
                num_key_value_heads=self.num_key_value_heads,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                dtype=self.dtype,
                layer_mapping=self.layer_mapping,
                prefill_bsz=self.prefill_bsz,
                num_gpus=self.num_gpus,
                model_size=int(re.search(r'(\d+)[B]', self.model_name).group(1))
            )
        elif self.attention_type == 'RetroInfer':
            retroinfer_config = config.get(self.attention_type)

            if retroinfer_config['gpu_only'] == True:
                self.kv_cache = retroinfer_cache_gpu(
                    valid_start=valid_start,
                    layer_num=self.num_layers,
                    batch_size=self.batch_size,
                    max_length=self.max_new_length + self.input_length,
                    num_key_value_heads=self.num_key_value_heads,
                    num_heads=self.num_heads,
                    head_dim=self.head_dim,
                    dtype=self.dtype,
                    layer_mapping=self.layer_mapping,
                    max_new_length=self.max_new_length,
                    static_pattern_start=retroinfer_config["static_pattern_start"],
                    static_pattern_end=retroinfer_config["static_pattern_end"],
                    core=retroinfer_config["core"],
                    n_centroids=retroinfer_config["n_centroids"],
                    n_segment=retroinfer_config["n_segment"],
                    pages_per_cluster=retroinfer_config["pages_per_cluster"],
                    retrieval_budget=retroinfer_config["retrieval_budget"],
                    estimation_budget=retroinfer_config["estimation_budget"],
                    buffer_cluster_num=retroinfer_config["buffer_cluster_num"],
                    prefill_bsz=self.prefill_bsz,
                    num_gpus=self.num_gpus,
                    model_size=int(re.search(r'(\d+)[B]', self.model_name).group(1))
                )
            else:
                self.kv_cache = retroinfer_cache(
                    valid_start=valid_start,
                    layer_num=self.num_layers,
                    batch_size=self.batch_size,
                    max_length=self.max_new_length + self.input_length,
                    num_key_value_heads=self.num_key_value_heads,
                    num_heads=self.num_heads,
                    head_dim=self.head_dim,
                    dtype=self.dtype,
                    layer_mapping=self.layer_mapping,
                    max_new_length=self.max_new_length,
                    static_pattern_start=retroinfer_config["static_pattern_start"],
                    static_pattern_end=retroinfer_config["static_pattern_end"],
                    core=retroinfer_config["core"],
                    n_centroids=retroinfer_config["n_centroids"],
                    n_segment=retroinfer_config["n_segment"],
                    pages_per_cluster=retroinfer_config["pages_per_cluster"],
                    retrieval_budget=retroinfer_config["retrieval_budget"],
                    estimation_budget=retroinfer_config["estimation_budget"],
                    cache_ratio=retroinfer_config["cache_ratio"],
                    buffer_cluster_num=retroinfer_config["buffer_cluster_num"],
                    use_cuda_graph=retroinfer_config["use_cuda_graph"],
                    prefill_bsz=self.prefill_bsz,
                    num_gpus=self.num_gpus,
                    model_size=int(re.search(r'(\d+)[B]', self.model_name).group(1))
                )
        else:
            raise ValueError(f"Unsupported attention type: {self.attention_type}")


    def move(self):
        torch.cuda.empty_cache()
        if self.attention_type == 'Full_Flash_Attn':
            self.kv_cache.move_gpu()
        elif self.attention_type == 'RetroInfer':
            self.kv_cache.prepare_cache()
        torch.cuda.empty_cache()


    def word_embedding(self, inputs_id):
        hidden_states = F.embedding(inputs_id, self.embed_tokens)
        return hidden_states


    def lm(self, hidden_states):
        logits = F.linear(hidden_states, self.lm_head).float()
        return logits


    def wqkv(self, hidden_states, layer):
        # No bias in Qwen3-MoE
        qkv = F.linear(hidden_states, layer.wqkv)
        # Split sizes differ from Qwen2.5 because head_dim*num_heads != hidden_size
        q_dim = self.num_heads * self.head_dim
        kv_dim = self.num_key_value_heads * self.head_dim
        query_states, key_states, value_states = qkv.split([q_dim, kv_dim, kv_dim], dim=-1)

        # Per-head QK RMSNorm (new in Qwen3)
        bsz, seq_len, _ = hidden_states.shape
        q_flat = query_states.reshape(bsz * seq_len * self.num_heads, self.head_dim)
        q_flat = flashinfer.rmsnorm(q_flat, layer.q_norm_weight, self.norm_eps)
        query_states = q_flat.reshape(bsz, seq_len, q_dim)

        k_flat = key_states.reshape(bsz * seq_len * self.num_key_value_heads, self.head_dim)
        k_flat = flashinfer.rmsnorm(k_flat, layer.k_norm_weight, self.norm_eps)
        key_states = k_flat.reshape(bsz, seq_len, kv_dim)

        return query_states, key_states, value_states


    def wo(self, hidden_states, layer, bsz, seq_len, dim):
        # Attention output dim = num_heads * head_dim (4096), not hidden_size (2048)
        attn_dim = self.num_heads * self.head_dim
        hidden_states = hidden_states.reshape(bsz, seq_len, attn_dim)
        hidden_states = F.linear(hidden_states, layer.wo)
        return hidden_states


    def prefill_attention(self, query_states, key_states, value_states, layer_idx):
        if self.prefill_method == "xattn":
            attn_out = prefill_xattn(query_states, key_states, value_states, self.thresholds[layer_idx], causal=True)
        elif self.prefill_method == "minfer":
            attn_out = prefill_minfer(query_states, key_states, value_states, self.best_patterns[layer_idx])
        else:
            attn_out = full_prefill_attn(query_states, key_states, value_states, causal=True)
        return attn_out


    def decode_attention(self, query_states, key_states, value_states, layer_idx):
        if self.attention_type == 'Full_Flash_Attn':
            attn_out = full_decode_attn(query_states, key_states, value_states, layer_idx, self.kv_cache)
        elif self.attention_type == 'RetroInfer':
            attn_out = retroinfer_decode_attn(query_states, key_states, value_states, layer_idx, self.kv_cache)
        else:
            raise ValueError(f"Unsupported attention type: {self.attention_type}")
        return attn_out


    def mlp(self, hidden_states, layer):
        """MoE MLP: route tokens to top-k experts, compute expert MLPs, aggregate."""
        orig_shape = hidden_states.shape
        hidden_dim = orig_shape[-1]
        hidden_states_flat = hidden_states.reshape(-1, hidden_dim)
        num_tokens = hidden_states_flat.shape[0]

        # Router: compute gating logits and select top-k experts
        router_logits = F.linear(hidden_states_flat, layer.gate_weight)  # [num_tokens, num_experts]
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        routing_weights, selected_experts = torch.topk(routing_weights, self.num_experts_per_tok, dim=-1)

        # Normalize routing weights to sum to 1
        if self.norm_topk_prob:
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        # Compute expert outputs via expert-parallel dispatch
        final_hidden_states = torch.zeros_like(hidden_states_flat)

        # Build expert assignment mask: [num_experts, top_k, num_tokens]
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts)
        expert_mask = expert_mask.permute(2, 1, 0)

        for expert_idx in range(self.num_experts):
            idx, top_x = torch.where(expert_mask[expert_idx])
            if top_x.shape[0] == 0:
                continue

            current_state = hidden_states_flat[top_x]
            # Gate-up projection + SiLU activation
            current_state = F.linear(current_state, layer.expert_gate_up_proj[expert_idx])
            gate_dim = current_state.shape[-1] // 2
            out = torch.empty(current_state.shape[:-1] + (gate_dim,),
                              dtype=current_state.dtype, device=current_state.device)
            flashinfer.activation.silu_and_mul(current_state, out)
            # Down projection
            current_state = F.linear(out, layer.expert_down_proj[expert_idx])

            final_hidden_states.index_add_(
                0, top_x, current_state * routing_weights[top_x, idx].unsqueeze(-1)
            )

        return final_hidden_states.reshape(orig_shape)


    def parameter_move(self, hidden_states, ldx):
        next_device = self.layer_mapping[str(ldx+1)] if str(ldx+1) in self.layer_mapping else self.layer_mapping[str(0)]
        torch.cuda.set_device(next_device)
        hidden_states = hidden_states.to(next_device)
        self.position_ids = self.position_ids.to(next_device)
        self.cos_sin_cache = self.cos_sin_cache.to(next_device)
        if self.attention_type == 'Full_Flash_Attn':
            if hidden_states.shape[1] == 1:
                self.kv_cache.batch_indices = self.kv_cache.batch_indices_dict[next_device]
                self.kv_cache.valid_length = self.kv_cache.valid_length_dict[next_device]
        elif self.attention_type == 'RetroInfer':
            if hidden_states.shape[1] == 1:
                if isinstance(self.kv_cache, retroinfer_cache_gpu):
                    self.kv_cache.gemm_o = self.kv_cache.gemm_o_dict[next_device]
                    self.kv_cache.softmax_o = self.kv_cache.softmax_o_dict[next_device]
                    self.kv_cache.norm = self.kv_cache.norm_dict[next_device]
                    self.kv_cache.sum = self.kv_cache.sum_dict[next_device]
                    self.kv_cache.dist = self.kv_cache.dist_dict[next_device]
                    self.kv_cache.cI = self.kv_cache.cI_dict[next_device]
                    self.kv_cache.cV = self.kv_cache.cV_dict[next_device]
                    self.kv_cache.es_centroids = self.kv_cache.es_centroids_dict[next_device]
                    self.kv_cache.es_value_sum = self.kv_cache.es_value_sum_dict[next_device]
                    self.kv_cache.es_cluster_size = self.kv_cache.es_cluster_size_dict[next_device]
                    self.kv_cache.execution_buffer_keys = self.kv_cache.execution_buffer_keys_dict[next_device]
                    self.kv_cache.execution_buffer_values = self.kv_cache.execution_buffer_values_dict[next_device]
                    self.kv_cache.valid_lengths = self.kv_cache.valid_lengths_dict[next_device]
                    self.kv_cache.static_len_tensor = self.kv_cache.static_len_tensor_dict[next_device]
                    self.kv_cache.nprobe_tensor = self.kv_cache.nprobe_tensor_dict[next_device]
                else:
                    self.kv_cache.cI = self.kv_cache.cI_dict[next_device]
                    self.kv_cache.static_len_tensor = self.kv_cache.static_len_tensor_dict[next_device]
                    if self.kv_cache.use_cuda_graph:
                        self.kv_cache.query_buffer = self.kv_cache.query_buffer_dict[next_device]
                        self.kv_cache.attn_out = self.kv_cache.attn_out_dict[next_device]
                    else:
                        self.kv_cache.gemm_o = self.kv_cache.gemm_o_dict[next_device]
                        self.kv_cache.softmax_o = self.kv_cache.softmax_o_dict[next_device]
                        self.kv_cache.norm = self.kv_cache.norm_dict[next_device]
                        self.kv_cache.sum = self.kv_cache.sum_dict[next_device]
                        self.kv_cache.dist = self.kv_cache.dist_dict[next_device]
                        self.kv_cache.cV = self.kv_cache.cV_dict[next_device]
                        self.kv_cache.es_centroids = self.kv_cache.es_centroids_dict[next_device]
                        self.kv_cache.es_value_sum = self.kv_cache.es_value_sum_dict[next_device]
                        self.kv_cache.es_cluster_size = self.kv_cache.es_cluster_size_dict[next_device]
                        self.kv_cache.execution_buffer_keys = self.kv_cache.execution_buffer_keys_dict[next_device]
                        self.kv_cache.execution_buffer_values = self.kv_cache.execution_buffer_values_dict[next_device]
                        self.kv_cache.valid_lengths = self.kv_cache.valid_lengths_dict[next_device]
        return hidden_states


    def layernorm(self, hidden_states, epsilon, weight):
        bsz, seq_len, dim = hidden_states.shape
        hidden_states = hidden_states.reshape(bsz * seq_len, dim)
        hidden_states = flashinfer.rmsnorm(hidden_states, weight, epsilon)
        hidden_states = hidden_states.reshape(bsz, seq_len, dim)
        return hidden_states


    def apply_rotary_pos_emb(self, query_states, key_states, position_ids):
        bsz, _, hidden_dim = query_states.shape
        _, _, kv_dim = key_states.shape
        query_states = query_states.view(-1, hidden_dim)
        key_states = key_states.view(-1, kv_dim)
        flashinfer.rope.apply_rope_with_cos_sin_cache_inplace(position_ids, query_states, key_states, self.head_dim, self.cos_sin_cache, True)
        query_states = query_states.view(bsz, -1, hidden_dim)
        key_states = key_states.view(bsz, -1, kv_dim)
        return query_states, key_states


    def position_embedd(self, query_states, key_states):
        bsz, seq_len, _ = key_states.shape
        position_ids = self.position_ids[self.kv_cache.context:self.kv_cache.context+seq_len].unsqueeze(0).repeat(bsz, 1)
        query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, position_ids)
        return query_states, key_states
