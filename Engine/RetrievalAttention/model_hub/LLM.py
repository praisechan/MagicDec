import time
import torch
from termcolor import colored


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
        

    def decode_forward(self, inputs_ids):
        hidden_states = self.word_embedding(inputs_ids)

        if self.num_gpus > 1:
            for ldx in range(self.num_layers):
                hidden_states = self.layer_decode(ldx, hidden_states)
                hidden_states = self.parameter_move(hidden_states, ldx)
            hidden_states = hidden_states.to(self.layers[0].device)
        else:
            for ldx in range(self.num_layers):
                # start_time = time.time()
                # torch.cuda.synchronize()
                hidden_states = self.layer_decode(ldx, hidden_states)
                # torch.cuda.synchronize()
                # end_time = time.time()
                # print(f"layer_decode:{end_time - start_time}")
        if self.profile_clustering:
            # profile only for first decoding step
            breakpoint()
        hidden_states = self.layernorm(hidden_states[:, -1:, :], self.norm_variance_epsilon, self.norm_weight)
        logits = self.lm(hidden_states)
        
        return logits


    def inference(self, inputs_ids):
        outputs_ids = []    # multi iteration, multi request
        output_ids = []     # single iteration, multi request
        
        outputs_logits = []
        top1_top2_diff = []
        print("Start prefilling ...")
        torch.cuda.synchronize()
        prefill_start = time.time()

        logits = self.prefill_forward(inputs_ids=inputs_ids)
        output_ids = logits.argmax(dim=-1)
        outputs_ids.append(output_ids)

        # make a list of top3 softmax value and its token id
        softmax_logits = torch.nn.functional.softmax(logits, dim=-1)
        topk_vals, topk_indices = torch.topk(softmax_logits, k=3, dim=-1)  # each is [B, 3]
        batch_top3 = []
        for i in range(3):
          batch_top3.append((topk_vals[0][:,i],topk_indices[0][:,i]))

        outputs_logits.append(batch_top3)
        top1_top2_diff.append(topk_vals[0][:,0]-topk_vals[0][:,1])
        self.move()

        torch.cuda.synchronize()
        prefill_end = time.time()
        print(colored(f"Prefilling latency: {round((prefill_end - prefill_start), 4)} s\n", 'green'))

        print("Start decoding ...")
        decode_start = time.time()

        hot_cluster_hit_ratio_per_layer = []
        hot_cluster_hit_ratio_per_token = []
        
        for _ in range(self.max_new_length-1):
            logits = self.decode_forward(inputs_ids=output_ids)
            output_ids = logits.argmax(dim=-1)
            outputs_ids.append(output_ids)
            
            softmax_logits = torch.nn.functional.softmax(logits, dim=-1)
            topk_vals, topk_indices = torch.topk(softmax_logits, k=3, dim=-1)  # each is [B, 3]
            batch_top3 = [[] for _ in range(topk_vals.shape[-2])]
            for i in range(topk_vals.shape[-2]):
              batch_top3[i].append((topk_vals[0][i][0],topk_indices[0][i][0]))
              batch_top3[i].append((topk_vals[0][i][1],topk_indices[0][i][1]))
              batch_top3[i].append((topk_vals[0][i][2],topk_indices[0][i][2]))
            
            outputs_logits.append(batch_top3)
            top1_top2_diff.append(topk_vals[0][:,0]-topk_vals[0][:,1])

            if self.attention_type == "RetroInfer":      
                # store hot cluster hit ratio
                hot_cluster_hit_ratio_per_layer.append(self.kv_cache.hot_cluster_hit_ratio.clone())
                hot_cluster_hit_ratio_per_token.append(self.kv_cache.hot_cluster_hit_ratio.mean())

        decode_end = time.time()
        print(colored(f"Decoding latency: {round((decode_end - decode_start), 8)} s\n", 'green'))

        # print(colored(
        #     f"Decoding latency: {round((decode_end - decode_start) * 1000 / (self.max_new_length - 1), 2)} ms/step, "
        #     f"Throughput: {round(self.batch_size * (self.max_new_length - 1) / (decode_end - decode_start), 2)} tokens/s\n",
        #     'green'
        # ))
        
        outputs_ids = torch.cat(outputs_ids, dim=-1).tolist()
        
        if self.attention_type == "RetroInfer" and self.profile_hot_clustering:
            window_size = self.kv_cache.window_size
            hot_cluster_ratio = self.kv_cache.hot_cluster_ratio
            cluster_size = self.kv_cache.avg_cluster_size
            budget_ratio = self.kv_cache.nprobe / self.kv_cache.n_centroids
            # hot cluster output
            filename = f"hot_cluster_input_{inputs_ids.shape[1]}_budget{budget_ratio}_hot{hot_cluster_ratio}_window{window_size}_cluster{cluster_size}.csv"
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

                for token_idx, data in enumerate(hot_cluster_hit_ratio_per_layer):
                    # Append one row per layer
                    for idx, val in enumerate(data):
                        v = val.item() if torch.is_tensor(val) else float(val)
                        writer.writerow([token_idx, idx, v])     
        
        return outputs_ids, outputs_logits, top1_top2_diff


    def generate(self, attention_type, inputs_ids, attention_masks, max_new_length, attn_config=None, profile_clustering=False, profile_hot_clustering=False):
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
        self.profile_hot_clustering = profile_hot_clustering

        valid_start = attention_masks.shape[1] - torch.sum(attention_masks, dim=-1).detach().cpu().numpy()
        del attention_masks
        torch.cuda.empty_cache()

        print("Allocate GPU buffers and CPU pin memory ...\n")
        self.init_kv_cache(input_length, valid_start, attn_config, profile_clustering=profile_clustering)

        outputs, logits, top1_top2_diff = self.inference(inputs_ids)

        return outputs, logits, top1_top2_diff
