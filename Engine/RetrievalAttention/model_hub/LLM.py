import time
from dataclasses import dataclass
from typing import Any, Optional

import torch
from termcolor import colored


@dataclass
class LLMSession:
    attention_type: str
    attention_masks: torch.Tensor
    batch_size: int
    input_length: int
    max_new_length: int
    valid_start: Any
    attn_config: Optional[dict] = None
    kv_cache: Any = None
    current_length: int = 0
    profile_clustering: bool = False
    profile_hot_cluster_selection_ratio: bool = False
    use_first_kv: bool = False
    gamma1: Optional[int] = None
    generate_name: Optional[str] = None


class LLM:
    """
    A class representing the LLM (currently support Llama and Qwen).
    """

    def __init__(
        self, 
        model_name: str,
        max_length: int,
        dtype: torch.dtype,
        device_map: str
    ) -> None:
        """ Initializes the LLM.
        Args:
            model_name (str): The name of the model.
            max_length (int): The maximum length (prefill+decode) of sequences.
            dtype (torch.dtype): The data type for model computations.
            device_map (str): The device for model, suppor 'cuda:x' or 'auto (automatically use all visible GPUs)'.
        """

        self.model_name = model_name
        self.max_length = max_length
        self.dtype = dtype
        self.device_map = device_map
        self.profile_clustering=False
        self.last_inference_profile = {}
        self.last_session_profile = {}


    def layer_prefill(self, layer_idx, start_bdx, hidden_states):
        # print(f'Layer = {layer_idx}, start_bdx = {start_bdx}')

        bsz, seq_len, dim = hidden_states.shape
        layer = self.layers[layer_idx]
        
        # original hidden_states used as residual, clone a new one to process
        temp_hidden_states = hidden_states.clone()

        # chunk for lower memory comsumption
        for start_idx in range(0, seq_len, 8192//bsz):
            end_idx = min(seq_len, start_idx + 8192//bsz)
            temp_hidden_states[:, start_idx:end_idx, :] = self.layernorm(temp_hidden_states[:, start_idx:end_idx, :], 
                                                                         layer.input_layernorm_variance_epsilon, 
                                                                         layer.input_layernorm_weight)
        
        query_states, key_states, value_states = self.wqkv(temp_hidden_states, layer)
        del temp_hidden_states
        torch.cuda.empty_cache()
        query_states, key_states = self.position_embedd(query_states, key_states)

        query_states = query_states.view(bsz, seq_len, self.num_heads, self.head_dim)       # reshape [bs, seq_len, dim] => [bs, seq_len, head, head_dim]
        key_states = key_states.view(bsz, seq_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, seq_len, self.num_key_value_heads, self.head_dim)

        import time

        # start_time = time.time()
        # torch.cuda.synchronize()
        key_states, value_states = self.kv_cache.prefill_update_kv_cache(query_states, key_states, value_states, layer_idx, start_bdx)
        # torch.cuda.synchronize()
        # end_time = time.time()
        # print(f"kv clutering time: {end_time - start_time}s")

        torch.cuda.empty_cache()

        temp_attn_out = self.prefill_attention(query_states, key_states, value_states)

        self.kv_cache.sync(layer_idx, start_bdx)

        del query_states, key_states, value_states
        torch.cuda.empty_cache()

        hidden_states += self.wo(temp_attn_out, layer, temp_attn_out.shape[0], seq_len, dim)
        del temp_attn_out
        torch.cuda.empty_cache()

        # post attention
        residual = hidden_states.clone()

        # chunk for lower memory comsumption
        for start_idx in range(0, seq_len, 8192//bsz):
            end_idx = min(seq_len, start_idx + 8192//bsz)
            hidden_states[:, start_idx:end_idx, :] = self.layernorm(hidden_states[:, start_idx:end_idx, :], 
                                                                    layer.post_attention_layernorm_variance_epsilon, 
                                                                    layer.post_attention_layernorm_weight)
            hidden_states[:, start_idx:end_idx, :] = self.mlp(hidden_states[:, start_idx:end_idx, :], layer)   
        
        hidden_states += residual

        del residual
        torch.cuda.empty_cache()
                                                                                                   
        return hidden_states


    def layer_decode(self, layer_idx, hidden_states):
        # print(f'Layer = {layer_idx}')

        residual = hidden_states
        bsz, seq_len, dim = hidden_states.shape
        layer = self.layers[layer_idx]

        hidden_states = self.layernorm(hidden_states, layer.input_layernorm_variance_epsilon, layer.input_layernorm_weight)
        
        query_states, key_states, value_states = self.wqkv(hidden_states, layer)
        query_states, key_states = self.position_embedd(query_states, key_states)

        query_states = query_states.view(bsz, -1, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, -1, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, -1, self.num_key_value_heads, self.head_dim)

        key_states, value_states = self.kv_cache.decode_update_kv_cache(key_states, value_states, layer_idx)
        # start_time = time.time()
        # torch.cuda.synchronize()
        attn_out = self.decode_attention(query_states, key_states, value_states, layer_idx)
        # torch.cuda.synchronize()
        # end_time = time.time()
        # print(f"attn_decode:{end_time - start_time}")
        hidden_states = self.wo(attn_out, layer, bsz, seq_len, dim)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layernorm(hidden_states, layer.post_attention_layernorm_variance_epsilon, layer.post_attention_layernorm_weight)
        hidden_states = self.mlp(hidden_states, layer)
        hidden_states = residual + hidden_states

        return hidden_states


    def prefill_forward(self, inputs_ids):
        bsz, seq_len = inputs_ids.shape
        device = inputs_ids.device

        last_hidden_states = torch.empty((bsz, 1, self.hidden_size), dtype=self.dtype, device=device)
        for start_bdx in range(0, bsz, 1):
            end_bdx = min(bsz, start_bdx + 1)
            hidden_states = self.word_embedding(inputs_ids[start_bdx:end_bdx])  # [1, seq_len, hidden_size]

            if self.num_gpus > 1:
                for ldx in range(self.num_layers):
                    hidden_states = self.layer_prefill(ldx, start_bdx, hidden_states)
                    hidden_states = self.parameter_move(hidden_states, ldx)
                    torch.cuda.empty_cache()
                last_hidden_states[start_bdx:end_bdx] = hidden_states[:, -1:, :].to(self.layers[0].device)
            else:
                for ldx in range(self.num_layers):
                    # start_time = time.time()
                    # torch.cuda.synchronize()
                    hidden_states = self.layer_prefill(ldx, start_bdx, hidden_states)
                    # torch.cuda.synchronize()
                    # end_time = time.time()
                    # print(f"layer_prefill:{end_time - start_time}")

                    torch.cuda.empty_cache()
                last_hidden_states[start_bdx:end_bdx] = hidden_states[:, -1:, :]
        
        last_hidden_states = self.layernorm(last_hidden_states.contiguous(), self.norm_variance_epsilon, self.norm_weight)
        logits = self.lm(last_hidden_states)
        
        return logits
        

    def decode_forward(self, inputs_ids, intermediate_output):
        hidden_states = self.word_embedding(inputs_ids)

        if self.num_gpus > 1:
            for ldx in range(self.num_layers):
                if intermediate_output:
                    self.kv_cache.store_decoding_data=True
                hidden_states = self.layer_decode(ldx, hidden_states)
                hidden_states = self.parameter_move(hidden_states, ldx)
            hidden_states = hidden_states.to(self.layers[0].device)
        else:
            for ldx in range(self.num_layers):
                # start_time = time.time()
                # torch.cuda.synchronize()
                if intermediate_output:
                    self.kv_cache.store_decoding_data=True
                hidden_states = self.layer_decode(ldx, hidden_states)
                # torch.cuda.synchronize()
                # end_time = time.time()
                # print(f"layer_decode:{end_time - start_time}")
        # if self.profile_clustering:
        #     # profile only for first decoding step
        #     raise ValueError("profile only for first decoding step")
        hidden_states = self.layernorm(hidden_states[:, -1:, :], self.norm_variance_epsilon, self.norm_weight)
        logits = self.lm(hidden_states)
        
        return logits

    def _prefill_logits_to_outputs(self, logits):
        output_ids = logits.argmax(dim=-1)
        softmax_logits = torch.nn.functional.softmax(logits, dim=-1)
        topk_vals, topk_indices = torch.topk(softmax_logits, k=3, dim=-1)
        batch_top3 = []
        for i in range(3):
            batch_top3.append((topk_vals[0][:, i], topk_indices[0][:, i]))

        return output_ids.tolist(), [logits], [topk_vals[0][:, 0] - topk_vals[0][:, 1]], [batch_top3]

    def _activate_session(self, session: LLMSession):
        self.batch_size = session.batch_size
        self.input_length = session.input_length
        self.max_new_length = session.max_new_length
        self.attention_type = session.attention_type
        self.profile_clustering = session.profile_clustering
        self.profile_hot_cluster_selection_ratio = session.profile_hot_cluster_selection_ratio
        self.generate_name = session.generate_name
        self.kv_cache = session.kv_cache

    def begin_session(
        self,
        attention_type,
        inputs_ids,
        attention_masks,
        max_new_length,
        attn_config=None,
        profile_clustering=False,
        profile_hot_cluster_selection_ratio=False,
        use_first_kv=False,
        gamma1=None,
        generate_name=None,
    ):
        bs, input_length = inputs_ids.shape
        assert input_length + max_new_length <= self.max_length, (
            f"Error: input_length({input_length}) + max_new_length({max_new_length}) exceeds max_length({self.max_length})"
        )

        valid_start = attention_masks.shape[1] - torch.sum(attention_masks, dim=-1).detach().cpu().numpy()
        session = LLMSession(
            attention_type=attention_type,
            attention_masks=attention_masks,
            batch_size=bs,
            input_length=input_length,
            max_new_length=max_new_length,
            valid_start=valid_start,
            attn_config=attn_config,
            current_length=input_length,
            profile_clustering=profile_clustering,
            profile_hot_cluster_selection_ratio=profile_hot_cluster_selection_ratio,
            use_first_kv=use_first_kv,
            gamma1=gamma1,
            generate_name=generate_name,
        )

        self.attention_type = attention_type
        self.batch_size = bs
        self.input_length = input_length
        self.max_new_length = max_new_length
        self.profile_clustering = profile_clustering
        self.profile_hot_cluster_selection_ratio = profile_hot_cluster_selection_ratio
        self.generate_name = generate_name

        self.init_kv_cache(
            input_length,
            valid_start,
            attn_config,
            profile_clustering=profile_clustering,
            use_first_kv=use_first_kv,
            gamma1=gamma1,
            generate_name=generate_name,
        )
        session.kv_cache = self.kv_cache

        torch.cuda.synchronize()
        prefill_start = time.time()
        logits = self.prefill_forward(inputs_ids=inputs_ids)
        torch.cuda.synchronize()
        prefill_end = time.time()
        outputs, logit_list, top1_top2_diff, top3_logits = self._prefill_logits_to_outputs(logits)
        self.move()
        self.last_session_profile = {
            "mode": "begin_session",
            "prefill_seconds": prefill_end - prefill_start,
            "decode_seconds": 0.0,
        }

        return session, outputs, logit_list, top1_top2_diff, top3_logits

    def update_session_attn_config(
        self,
        session: LLMSession,
        attn_config=None,
        use_first_kv=None,
        gamma1=None,
        generate_name=None,
    ):
        if attn_config is not None:
            session.attn_config = attn_config
        if use_first_kv is not None:
            session.use_first_kv = use_first_kv
        if gamma1 is not None:
            session.gamma1 = gamma1
        if generate_name is not None:
            session.generate_name = generate_name

        if session.kv_cache is None:
            raise RuntimeError("Cannot update attention config for a session without a KV cache.")

        self._activate_session(session)
        retroinfer_config = None
        if session.attention_type == "RetroInfer" and session.attn_config is not None:
            retroinfer_config = session.attn_config.get("RetroInfer")

        if hasattr(session.kv_cache, "update_runtime_config"):
            session.kv_cache.update_runtime_config(
                retroinfer_config,
                use_first_kv=session.use_first_kv,
                gamma1=session.gamma1,
                generate_name=session.generate_name,
            )

    def snapshot_session(self, session: LLMSession):
        if session.kv_cache is None:
            raise RuntimeError("Cannot snapshot a session before KV cache initialization.")
        if not hasattr(session.kv_cache, "snapshot_state"):
            raise RuntimeError(f"KV cache for {session.attention_type} does not support snapshot_state().")

        return {
            "current_length": session.current_length,
            "kv_cache": session.kv_cache.snapshot_state(),
        }

    def restore_session(self, session: LLMSession, snapshot):
        if snapshot is None:
            raise ValueError("snapshot must be provided for restore_session.")
        if session.kv_cache is None:
            raise RuntimeError("Cannot restore a session before KV cache initialization.")
        if "kv_cache" not in snapshot or "current_length" not in snapshot:
            raise ValueError("Invalid session snapshot.")

        self._activate_session(session)
        session.kv_cache.restore_state(snapshot["kv_cache"])
        session.current_length = int(snapshot["current_length"])

    def decode_session(
        self,
        session: LLMSession,
        bonus_token,
        num_new_tokens,
        use_first_kv=None,
        gamma1=None,
        generate_name=None,
    ):
        if bonus_token is None:
            raise ValueError("bonus_token must be provided for decode_session.")
        if num_new_tokens < 0:
            raise ValueError(f"num_new_tokens must be non-negative, got {num_new_tokens}.")
        if num_new_tokens == 0:
            return [[] for _ in range(session.batch_size)], [], [], []

        if use_first_kv is not None:
            session.use_first_kv = use_first_kv
        if gamma1 is not None:
            session.gamma1 = gamma1
        if generate_name is not None:
            session.generate_name = generate_name

        self._activate_session(session)
        if hasattr(session.kv_cache, "begin_decode_call"):
            session.kv_cache.begin_decode_call(
                use_first_kv=session.use_first_kv,
                gamma1=session.gamma1,
                generate_name=session.generate_name,
            )

        output_ids_list = []
        top3_logits = []
        top1_top2_diff = []
        logit_list = []
        output_ids = bonus_token

        torch.cuda.synchronize()
        decode_start = time.time()
        for step in range(num_new_tokens):
            self.kv_cache.decoding_step = step

            intermediate_output = False
            if self.profile_clustering:
                if self.generate_name and "verify" in self.generate_name:
                    if step == 0:
                        intermediate_output = True
                else:
                    intermediate_output = True

            logits = self.decode_forward(inputs_ids=output_ids, intermediate_output=intermediate_output)
            output_ids = logits.argmax(dim=-1)
            output_ids_list.append(output_ids)

            softmax_logits = torch.nn.functional.softmax(logits, dim=-1)
            topk_vals, topk_indices = torch.topk(softmax_logits, k=3, dim=-1)
            batch_top3 = [[] for _ in range(topk_vals.shape[-2])]
            for i in range(topk_vals.shape[-2]):
                batch_top3[i].append((topk_vals[0][i][0], topk_indices[0][i][0]))
                batch_top3[i].append((topk_vals[0][i][1], topk_indices[0][i][1]))
                batch_top3[i].append((topk_vals[0][i][2], topk_indices[0][i][2]))

            top3_logits.append(batch_top3)
            top1_top2_diff.append(topk_vals[0][:, 0] - topk_vals[0][:, 1])
            logit_list.append(logits)

        torch.cuda.synchronize()
        decode_end = time.time()
        session.current_length += num_new_tokens
        self.last_session_profile = {
            "mode": "decode_session",
            "prefill_seconds": 0.0,
            "decode_seconds": decode_end - decode_start,
        }
        output_ids_list = torch.cat(output_ids_list, dim=-1).tolist()
        return output_ids_list, logit_list, top1_top2_diff, top3_logits

    def append_tokens(self, session: LLMSession, tokens, generate_name=None):
        if tokens is None:
            raise ValueError("tokens must be provided for append_tokens.")
        if tokens.ndim != 2:
            raise ValueError(f"tokens must have shape [batch, seq], got {tuple(tokens.shape)}.")
        if tokens.shape[1] == 0:
            return

        self._activate_session(session)
        if generate_name is not None:
            session.generate_name = generate_name
            self.generate_name = generate_name
        if hasattr(session.kv_cache, "begin_decode_call"):
            session.kv_cache.begin_decode_call(use_first_kv=False, gamma1=None, generate_name=session.generate_name)

        for step in range(tokens.shape[1]):
            self.kv_cache.decoding_step = step
            self.decode_forward(inputs_ids=tokens[:, step:step+1], intermediate_output=False)

        session.current_length += tokens.shape[1]

    def end_session(self, session: LLMSession):
        if session is None or session.kv_cache is None:
            return

        kv_cache = session.kv_cache
        session.kv_cache = None
        if self.kv_cache is kv_cache:
            self.kv_cache = None
        del kv_cache


    def inference(self, inputs_ids):
        output_ids_list = []    # multi iteration, multi request
        output_ids = []     # single iteration, multi request
        
        top3_logits = []
        top1_top2_diff = []
        logit_list = []

        print("Start prefilling ...")
        torch.cuda.synchronize()
        prefill_start = time.time()

        logits = self.prefill_forward(inputs_ids=inputs_ids)
        output_ids = logits.argmax(dim=-1)
        output_ids_list.append(output_ids)

        # make a list of top3 softmax value and its token id
        softmax_logits = torch.nn.functional.softmax(logits, dim=-1)
        topk_vals, topk_indices = torch.topk(softmax_logits, k=3, dim=-1)  # each is [B, 3]
        batch_top3 = []
        for i in range(3):
          batch_top3.append((topk_vals[0][:,i],topk_indices[0][:,i]))

        top3_logits.append(batch_top3)
        top1_top2_diff.append(topk_vals[0][:,0]-topk_vals[0][:,1])
        logit_list.append(logits)
        self.move()

        torch.cuda.synchronize()
        prefill_end = time.time()
        print(colored(f"Prefilling latency: {round((prefill_end - prefill_start), 4)} s\n", 'green'))

        print("Start decoding ...")
        decode_start = time.time()

        hot_cluster_hit_ratio_per_layer = []
        hot_cluster_hit_ratio_per_token = []

        # profile_decoding_steps = [0, 128, 256, 512, 1022]
        # intermediate_output = False
  
        for step in range(self.max_new_length-1):          
            # flag kv_cache to store profile data
            self.kv_cache.decoding_step = step

            # if step in profile_decoding_steps:
            #     intermediate_output = True
            intermediate_output = False
            if self.profile_clustering:
                if self.generate_name and "verify" in self.generate_name:
                    # verify stage only needs the first token's kv cache
                    if step == 0:
                        intermediate_output = True
                else:
                    intermediate_output = True
                          
            logits = self.decode_forward(inputs_ids=output_ids, intermediate_output=intermediate_output)
            output_ids = logits.argmax(dim=-1)
            output_ids_list.append(output_ids)
            
            # for output token probabiltiy profile
            softmax_logits = torch.nn.functional.softmax(logits, dim=-1)
            topk_vals, topk_indices = torch.topk(softmax_logits, k=3, dim=-1)  # each is [B, 3]
            batch_top3 = [[] for _ in range(topk_vals.shape[-2])]
            for i in range(topk_vals.shape[-2]):
              batch_top3[i].append((topk_vals[0][i][0],topk_indices[0][i][0]))
              batch_top3[i].append((topk_vals[0][i][1],topk_indices[0][i][1]))
              batch_top3[i].append((topk_vals[0][i][2],topk_indices[0][i][2]))
            
            top3_logits.append(batch_top3)
            top1_top2_diff.append(topk_vals[0][:,0]-topk_vals[0][:,1])
            logit_list.append(logits)

            # store hot cluster hit ratio
            if self.attention_type == "RetroInfer" and self.profile_hot_cluster_selection_ratio:
                hot_cluster_hit_ratio_per_layer.append(self.kv_cache.hot_cluster_hit_ratio.clone())
                hot_cluster_hit_ratio_per_token.append(self.kv_cache.hot_cluster_hit_ratio.mean())

        decode_end = time.time()
        print(colored(f"Decoding latency: {round((decode_end - decode_start), 8)} s\n", 'green'))
        self.last_inference_profile = {
            "mode": "generate",
            "prefill_seconds": prefill_end - prefill_start,
            "decode_seconds": decode_end - decode_start,
        }

        # print(colored(
        #     f"Decoding latency: {round((decode_end - decode_start) * 1000 / (self.max_new_length - 1), 2)} ms/step, "
        #     f"Throughput: {round(self.batch_size * (self.max_new_length - 1) / (decode_end - decode_start), 2)} tokens/s\n",
        #     'green'
        # ))
        
        output_ids_list = torch.cat(output_ids_list, dim=-1).tolist()
        
        if self.attention_type == "RetroInfer" and self.profile_hot_cluster_selection_ratio:
            window_size = self.kv_cache.window_size
            hot_cluster_ratio = self.kv_cache.hot_cluster_ratio
            cluster_size = self.kv_cache.avg_cluster_size
            budget_ratio = self.kv_cache.nprobe / self.kv_cache.n_centroids
            # hot cluster output
            filename = f"output/hot_cluster_input_{inputs_ids.shape[1]}_budget{budget_ratio}_hot{hot_cluster_ratio}_window{window_size}_cluster{cluster_size}.csv"
            # Check whether the file already exists
            import os
            import csv
            file_exists = os.path.isfile(filename)

            # Open in append mode
            with open(filename, 'a', newline='') as f:
                writer = csv.writer(f)
                # If the file is new, write the header
                if not file_exists:
                    writer.writerow(['token_num','layer_idx', 'hit_ratio'])

                for token_idx, data in enumerate(hot_cluster_hit_ratio_per_token):
                    v = data.item() if torch.is_tensor(data) else float(data)
                    writer.writerow([token_idx, "total", v])     

                # for token_idx, data in enumerate(hot_cluster_hit_ratio_per_layer):
                #     # Append one row per layer
                #     for idx, val in enumerate(data):
                #         v = val.item() if torch.is_tensor(val) else float(val)
                #         writer.writerow([token_idx, idx, v])     
        
        return output_ids_list, top3_logits, top1_top2_diff, logit_list

    def inference_without_prefill_token(self, inputs_ids, bonus_token):
        if bonus_token is None:
            raise ValueError("bonus_token must be provided for inference_without_prefill_token")

        output_ids_list = []    # multi iteration, multi request
        output_ids = []     # single iteration, multi request
        
        top3_logits = []
        top1_top2_diff = []
        logit_list = []

        print("Start prefilling ...")
        torch.cuda.synchronize()
        prefill_start = time.time()

        _ = self.prefill_forward(inputs_ids=inputs_ids)
        self.move()

        torch.cuda.synchronize()
        prefill_end = time.time()
        print(colored(f"Prefilling latency: {round((prefill_end - prefill_start), 4)} s\n", 'green'))

        print("Start decoding ...")
        decode_start = time.time()

        hot_cluster_hit_ratio_per_layer = []
        hot_cluster_hit_ratio_per_token = []

        # profile_decoding_steps = [0, 128, 256, 512, 1022]
        # intermediate_output = False

        output_ids = bonus_token

        for step in range(self.max_new_length-1):          
            # flag kv_cache to store profile data
            self.kv_cache.decoding_step = step

            # if step in profile_decoding_steps:
            #     intermediate_output = True

            intermediate_output = False
            if self.profile_clustering:
                if self.generate_name and "verify" in self.generate_name:
                    # verify stage only needs the first token's kv cache
                    if step == 0:
                        intermediate_output = True
                else:
                    intermediate_output = True
                          
            logits = self.decode_forward(inputs_ids=output_ids, intermediate_output=intermediate_output) # use bonus token as the first token
            output_ids = logits.argmax(dim=-1)
            output_ids_list.append(output_ids)
            
            # for output token probabiltiy profile
            softmax_logits = torch.nn.functional.softmax(logits, dim=-1)
            topk_vals, topk_indices = torch.topk(softmax_logits, k=3, dim=-1)  # each is [B, 3]
            batch_top3 = [[] for _ in range(topk_vals.shape[-2])]
            for i in range(topk_vals.shape[-2]):
              batch_top3[i].append((topk_vals[0][i][0],topk_indices[0][i][0]))
              batch_top3[i].append((topk_vals[0][i][1],topk_indices[0][i][1]))
              batch_top3[i].append((topk_vals[0][i][2],topk_indices[0][i][2]))
            
            top3_logits.append(batch_top3)
            top1_top2_diff.append(topk_vals[0][:,0]-topk_vals[0][:,1])
            logit_list.append(logits)

            # store hot cluster hit ratio
            if self.attention_type == "RetroInfer" and self.profile_hot_cluster_selection_ratio:
                hot_cluster_hit_ratio_per_layer.append(self.kv_cache.hot_cluster_hit_ratio.clone())
                hot_cluster_hit_ratio_per_token.append(self.kv_cache.hot_cluster_hit_ratio.mean())

        decode_end = time.time()
        print(colored(f"Decoding latency: {round((decode_end - decode_start), 8)} s\n", 'green'))
        self.last_inference_profile = {
            "mode": "generate_without_prefill_token",
            "prefill_seconds": prefill_end - prefill_start,
            "decode_seconds": decode_end - decode_start,
        }

        # print(colored(
        #     f"Decoding latency: {round((decode_end - decode_start) * 1000 / (self.max_new_length - 1), 2)} ms/step, "
        #     f"Throughput: {round(self.batch_size * (self.max_new_length - 1) / (decode_end - decode_start), 2)} tokens/s\n",
        #     'green'
        # ))
        
        output_ids_list = torch.cat(output_ids_list, dim=-1).tolist()
        
        if self.attention_type == "RetroInfer" and self.profile_hot_cluster_selection_ratio:
            window_size = self.kv_cache.window_size
            hot_cluster_ratio = self.kv_cache.hot_cluster_ratio
            cluster_size = self.kv_cache.avg_cluster_size
            budget_ratio = self.kv_cache.nprobe / self.kv_cache.n_centroids
            # hot cluster output
            filename = f"output/hot_cluster_input_{inputs_ids.shape[1]}_budget{budget_ratio}_hot{hot_cluster_ratio}_window{window_size}_cluster{cluster_size}.csv"
            # Check whether the file already exists
            import os
            import csv
            file_exists = os.path.isfile(filename)

            # Open in append mode
            with open(filename, 'a', newline='') as f:
                writer = csv.writer(f)
                # If the file is new, write the header
                if not file_exists:
                    writer.writerow(['token_num','layer_idx', 'hit_ratio'])

                for token_idx, data in enumerate(hot_cluster_hit_ratio_per_token):
                    v = data.item() if torch.is_tensor(data) else float(data)
                    writer.writerow([token_idx, "total", v])     

                # for token_idx, data in enumerate(hot_cluster_hit_ratio_per_layer):
                #     # Append one row per layer
                #     for idx, val in enumerate(data):
                #         v = val.item() if torch.is_tensor(val) else float(val)
                #         writer.writerow([token_idx, idx, v])     
        
        return output_ids_list, top3_logits, top1_top2_diff, logit_list


    def generate(self, attention_type, inputs_ids, attention_masks, max_new_length, attn_config=None, profile_clustering=False, profile_hot_cluster_selection_ratio=False, use_first_kv=False, gamma1=None, generate_name=None):
        """ LLM Inference.
        Args:
            attention_type: str,
            input_ids (torch.tensor): The input of LLM.
            attention_masks (torch.tensor): The attention masks of LLM.
            max_new_length (int): The maximum length of generated sequences.
        """

        bs, input_length = inputs_ids.shape
        assert input_length + max_new_length <= self.max_length, \
        f"Error: input_length({input_length}) + max_new_length({max_new_length}) exceeds max_length({self.max_length})"

        self.batch_size = bs
        self.input_length = input_length
        self.max_new_length = max_new_length
        self.attention_type = attention_type
        self.profile_clustering = profile_clustering
        self.profile_hot_cluster_selection_ratio = profile_hot_cluster_selection_ratio
        self.generate_name = generate_name

        valid_start = attention_masks.shape[1] - torch.sum(attention_masks, dim=-1).detach().cpu().numpy()
        del attention_masks
        torch.cuda.empty_cache()

        print("Allocate GPU buffers and CPU pin memory ...\n")
        self.init_kv_cache(input_length, valid_start, attn_config, profile_clustering=profile_clustering, use_first_kv=use_first_kv, gamma1=gamma1, generate_name=generate_name)

        outputs, top3_logits, top1_top2_diff, logit_list = self.inference(inputs_ids)

        return outputs, logit_list, top1_top2_diff, top3_logits

    def generate_without_prefill_token(self, attention_type, inputs_ids, bonus_token, attention_masks, max_new_length, attn_config=None, profile_clustering=False, profile_hot_cluster_selection_ratio=False, use_first_kv=False, gamma1=None, generate_name=None):
        """ LLM Inference.
        Args:
            attention_type: str,
            input_ids (torch.tensor): The input of LLM. Different from generate(), this input does not include the bonus token.
            bonus_token (torch.tensor): The bonus token to be used as the first token in the generation.
            attention_masks (torch.tensor): The attention masks of LLM.
            max_new_length (int): The maximum length of generated sequences.
        """

        bs, input_length = inputs_ids.shape
        assert input_length + max_new_length <= self.max_length, \
        f"Error: input_length({input_length}) + max_new_length({max_new_length}) exceeds max_length({self.max_length})"

        self.batch_size = bs
        self.input_length = input_length
        self.max_new_length = max_new_length
        self.attention_type = attention_type
        self.profile_clustering = profile_clustering
        self.profile_hot_cluster_selection_ratio = profile_hot_cluster_selection_ratio
        self.generate_name = generate_name

        valid_start = attention_masks.shape[1] - torch.sum(attention_masks, dim=-1).detach().cpu().numpy()
        del attention_masks
        torch.cuda.empty_cache()

        print("Allocate GPU buffers and CPU pin memory ...\n")
        self.init_kv_cache(input_length, valid_start, attn_config, profile_clustering=profile_clustering, use_first_kv=use_first_kv, gamma1=gamma1, generate_name=generate_name)

        outputs, top3_logits, top1_top2_diff, logit_list = self.inference_without_prefill_token(inputs_ids, bonus_token=bonus_token)

        return outputs, logit_list, top1_top2_diff, top3_logits

    def cleanup_kv_cache(self):
        """Clean up KV cache to free GPU memory."""
        if hasattr(self, 'kv_cache') and self.kv_cache is not None:
            # Delete key and value caches if they exist
            if hasattr(self.kv_cache, 'key_cache') and self.kv_cache.key_cache is not None:
                for cache in self.kv_cache.key_cache:
                    if cache is not None:
                        del cache
                del self.kv_cache.key_cache
                self.kv_cache.key_cache = None
            
            if hasattr(self.kv_cache, 'value_cache') and self.kv_cache.value_cache is not None:
                for cache in self.kv_cache.value_cache:
                    if cache is not None:
                        del cache
                del self.kv_cache.value_cache
                self.kv_cache.value_cache = None
            
            # Delete any other cache-related tensors
            if hasattr(self.kv_cache, 'cluster_centers') and self.kv_cache.cluster_centers is not None:
                for centers in self.kv_cache.cluster_centers:
                    if centers is not None:
                        del centers
                del self.kv_cache.cluster_centers
                self.kv_cache.cluster_centers = None
            
            # Delete the cache object itself
            del self.kv_cache
            self.kv_cache = None
        
        # Force garbage collection and empty CUDA cache
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        print("KV cache cleaned up successfully")
