import math
import torch
from retroinfer_kernels import ThreadPool, WaveBufferCPU
from retroinfer_kernels import gather_copy_and_concat, gather_copy_and_scatter, gather_copy_vectors, batch_gemm_softmax

from .cache import KV_Cache
from .kmeans import segment_k_means
from weighted_flash_decoding import weighted_flash_decoding


# update segment size
THRESHOLD_LENGTH = 1024


class retroinfer_cache(KV_Cache):
    """
    A class representing the KV Cache of RetroInfer.
    """

    def __init__(
        self,
        valid_start,
        layer_num: int,
        batch_size: int,
        max_length: int,
        num_key_value_heads: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        layer_mapping: dict,
        max_new_length: int,
        static_pattern_start: int,
        static_pattern_end: int,
        core: int,
        n_centroids: int,
        n_segment: int,
        nprobe: int,
        max_compute_cluster_num: int,
        cache_unit_size: int,
        cache_cluster_num: int,
        num_gpus: int,
        model_size: int,
        profile_clustering: bool
    ) -> None:
        super().__init__(layer_num, batch_size, max_length, num_key_value_heads, num_heads, head_dim, dtype, layer_mapping, num_gpus, model_size)
        self.valid_start = valid_start

        self.static_pattern_start = static_pattern_start
        self.static_pattern_end = static_pattern_end
        self.static_pattern_total = self.static_pattern_start + self.static_pattern_end

        self.group_size = self.num_heads // self.kv_head
        self.batch_groups = self.batch_size * self.kv_head

        self.page_size = cache_unit_size

        self.core = core
        self.dtype = dtype

        self.input_length = self.max_length - max_new_length
        self.max_new_length = min(max_new_length-1, THRESHOLD_LENGTH)   # already generated one token when prefilling
        # used for index update, when exceed THRESHOLD_LENGTH, we need to update the index
        self.input_length_new = ((max_new_length-2) // THRESHOLD_LENGTH) * THRESHOLD_LENGTH
        self.n_centroids_per_update_segment = THRESHOLD_LENGTH // 16    # default avg 16 vectors per cluster
        self.n_centroids_per_update_segment = (self.n_centroids_per_update_segment // 32) * 32 # must be divisible by 32
        self.n_centroids_new = ((max_new_length-2) // THRESHOLD_LENGTH) * self.n_centroids_per_update_segment  
        if self.input_length_new > 0:
            self.offload_update_keys = torch.empty(
                (self.batch_size*self.kv_head, THRESHOLD_LENGTH, self.head_dim), dtype=self.dtype, pin_memory=True
            ).contiguous()
            self.offload_update_values = torch.empty(
                (self.batch_size*self.kv_head, THRESHOLD_LENGTH, self.head_dim), dtype=self.dtype, pin_memory=True
            ).contiguous()

        # constant values
        self.RSQRT_DIM = 1.0 / math.sqrt(self.head_dim)
        self.DTYPE_MIN = torch.finfo(self.dtype).min

        # store steady zone
        self.steady_zone_keys = [
            torch.zeros((self.batch_size, self.kv_head, self.static_pattern_total+self.max_new_length, self.head_dim), 
            dtype=self.dtype, device=self.layer_mapping[str(ldx)]
            ) for ldx in range(self.layer_num)
        ]
        self.steady_zone_values = [
            torch.zeros((self.batch_size, self.kv_head, self.static_pattern_total+self.max_new_length, self.head_dim), 
            dtype=self.dtype, device=self.layer_mapping[str(ldx)]
            ) for ldx in range(self.layer_num)
        ]
        self.static_stride = self.static_pattern_total + self.max_new_length

        # index parameters
        self.n_centroids = n_centroids
        self.n_segment = n_segment
        self.approx_supercluster_size = 16
        self.n_super_centroids = int(n_centroids/self.approx_supercluster_size)
        self.nprobe = nprobe    # retrieve zone size
        self.max_compute_cluster_num = max_compute_cluster_num
        self.es_cluster_num = max_compute_cluster_num - nprobe  # estimation zone size

        # initialize thread pool
        self.thread_pool = ThreadPool(core)
        thread_pool_pointer = self.thread_pool.get()

        # calculate the gpu cache size, buffer size and max total pages for each group
        avg_cluster_size = (self.input_length - self.static_pattern_total) // self.n_centroids
        pages_per_cluster = math.ceil(avg_cluster_size / self.page_size)
        self.cache_size = cache_cluster_num * pages_per_cluster
        # enlarge these values may solve warning and error when decoding
        self.buffer_size = max(int(self.nprobe * 4), 16) * pages_per_cluster

        # whether to pre-allocate GPU buffer and cache before prefilling
        self.allocated = self.pre_allocate_decision()

        # initialize the CPU Wave Buffer
        self.wave_buffer = [WaveBufferCPU(
            self.batch_size, self.kv_head, self.head_dim, self.nprobe, self.page_size, self.n_centroids, 
            self.n_centroids+self.n_centroids_new, self.buffer_size, self.cache_size, self.core, thread_pool_pointer)
            for _ in range(self.layer_num)
        ]

        # pin memory indices for hit clusters
        self.hit_unit_idices = [
            torch.zeros((self.batch_size*self.kv_head, self.buffer_size), dtype=torch.int32, pin_memory=True).contiguous()
            for _ in range(self.layer_num)
        ]
        self.hit_unit_sizes = [
            torch.zeros((self.batch_size*self.kv_head, self.buffer_size), dtype=torch.int32, pin_memory=True).contiguous()
            for _ in range(self.layer_num)
        ]
        self.hit_unit_sizes_cumsum = [
            torch.zeros((self.batch_size*self.kv_head, self.buffer_size), dtype=torch.int32, pin_memory=True).contiguous()
            for _ in range(self.layer_num)
        ]
        self.hit_num_units = [
            torch.zeros((self.batch_size*self.kv_head), dtype=torch.int32, pin_memory=True).contiguous()
            for _ in range(self.layer_num)
        ]
        # pin memory indices for missing clusters
        self.miss_unit_idices = [
            torch.zeros((self.batch_size*self.kv_head, self.buffer_size), dtype=torch.int32, pin_memory=True).contiguous()
            for _ in range(self.layer_num)
        ]
        self.miss_unit_sizes = [
            torch.zeros((self.batch_size*self.kv_head, self.buffer_size), dtype=torch.int32, pin_memory=True).contiguous()
            for _ in range(self.layer_num)
        ]
        self.miss_unit_sizes_cumsum = [
            torch.zeros((self.batch_size*self.kv_head, self.buffer_size), dtype=torch.int32, pin_memory=True).contiguous()
            for _ in range(self.layer_num)
        ]
        self.miss_num_units = [
            torch.zeros((self.batch_size*self.kv_head), dtype=torch.int32, pin_memory=True).contiguous()
            for _ in range(self.layer_num)
        ]
        # pin memory indices for cache update clusters
        self.update_buffer_indices = [
            torch.zeros((self.batch_size*self.kv_head, self.buffer_size), dtype=torch.int32, pin_memory=True).contiguous()
            for _ in range(self.layer_num)
        ]
        self.update_unit_sizes = [
            torch.zeros((self.batch_size*self.kv_head, self.buffer_size), dtype=torch.int32, pin_memory=True).contiguous()
            for _ in range(self.layer_num)
        ]
        self.update_cache_indices = [
            torch.zeros((self.batch_size*self.kv_head, self.buffer_size), dtype=torch.int32, pin_memory=True).contiguous()
            for _ in range(self.layer_num)
        ]
        self.update_num_units = [
            torch.zeros((self.batch_size*self.kv_head), dtype=torch.int32, pin_memory=True).contiguous()
            for _ in range(self.layer_num)
        ]

        # store searched topk cluster ids
        self.cluster_ids = torch.empty((self.batch_size*self.kv_head, self.nprobe), dtype=torch.int64, pin_memory=True).contiguous()

        for ldx in range(self.layer_num):
            self.wave_buffer[ldx].set_indices(
                self.hit_unit_idices[ldx], self.hit_unit_sizes[ldx], self.hit_unit_sizes_cumsum[ldx], self.hit_num_units[ldx],
                self.miss_unit_idices[ldx], self.miss_unit_sizes[ldx], self.miss_unit_sizes_cumsum[ldx], self.miss_num_units[ldx],
                self.update_buffer_indices[ldx], self.update_unit_sizes[ldx], self.update_cache_indices[ldx], self.update_num_units[ldx], 
                self.cluster_ids
            )

        if self.allocated:
            self.cache_keys = []
            self.cache_values = []
            self.centroids = []
            self.value_sum = []
            self.centroids_mask = []
            self.cluster_size = []
            # allocate GPU Cache data and meta index
            for ldx in range(self.layer_num):
                self.cache_keys.append(
                    torch.zeros((self.batch_size, self.kv_head, self.cache_size, self.page_size, self.head_dim),
                                dtype=self.dtype, device=self.layer_mapping[str(ldx)]).contiguous()
                )
                self.cache_values.append(
                    torch.zeros((self.batch_size, self.kv_head, self.cache_size, self.page_size, self.head_dim),
                                dtype=self.dtype, device=self.layer_mapping[str(ldx)]).contiguous()
                )
                self.centroids.append(
                    torch.zeros((self.batch_size*self.kv_head, self.n_centroids, self.head_dim), 
                                dtype=self.dtype, device=self.layer_mapping[str(ldx)]).contiguous()
                )
                self.value_sum.append(
                    torch.zeros((self.batch_size*self.kv_head, self.n_centroids, self.head_dim), 
                                dtype=self.dtype, device=self.layer_mapping[str(ldx)]).contiguous()
                )
                self.centroids_mask.append(
                    torch.zeros((self.batch_size*self.kv_head, self.n_centroids), 
                                dtype=torch.bool, device=self.layer_mapping[str(ldx)]).contiguous()
                )
                self.cluster_size.append(
                    torch.zeros((self.batch_size*self.kv_head, self.n_centroids),
                                dtype=self.dtype, device=self.layer_mapping[str(ldx)]).contiguous()
                )
            self.cache_stride = self.cache_size
            self.allocate_computation_buffer()
        else:
            # allocate meta index in CPU
            self.centroids = [
                torch.zeros((self.batch_size*self.kv_head, self.n_centroids, self.head_dim), 
                            dtype=self.dtype, device="cpu").contiguous()
                for ldx in range(self.layer_num)
            ]
            self.value_sum = [
                torch.zeros((self.batch_size*self.kv_head, self.n_centroids, self.head_dim), 
                            dtype=self.dtype, device="cpu").contiguous()
                for ldx in range(self.layer_num)
            ]
            self.centroids_mask = [
                torch.zeros((self.batch_size*self.kv_head, self.n_centroids), 
                            dtype=torch.bool, device="cpu").contiguous()
                for ldx in range(self.layer_num)
            ]
            self.cluster_size = [
                torch.zeros((self.batch_size*self.kv_head, self.n_centroids),
                            dtype=self.dtype, device="cpu").contiguous()
                for ldx in range(self.layer_num)
            ]

        # layer-share cpu pin buffer, transfer gpu keys & values to cpu for segmented k-means
        self.offload_keys = torch.empty(
            (self.kv_head, self.input_length-self.static_pattern_total, self.head_dim), 
            dtype=self.dtype, pin_memory=True
        ).contiguous()
        self.offload_values = torch.empty(
            (self.kv_head, self.input_length-self.static_pattern_total, self.head_dim), 
            dtype=self.dtype, pin_memory=True
        ).contiguous()

        # allocate pin memory to store organized keys & values in CPU
        self.list_keys = []
        self.list_values = []
        for _ in range(self.layer_num):
            self.list_keys.append(
                torch.empty((self.batch_size, self.kv_head, self.input_length-self.static_pattern_total+self.input_length_new, self.head_dim), 
                            dtype=self.dtype, pin_memory=True).contiguous()
            )
            self.list_values.append(
                torch.empty((self.batch_size, self.kv_head, self.input_length-self.static_pattern_total+self.input_length_new, self.head_dim),
                            dtype=self.dtype, pin_memory=True).contiguous()
            )
        self.list_stride = self.input_length-self.static_pattern_total+self.input_length_new
        for ldx in range(self.layer_num):
            self.wave_buffer[ldx].set_kv(self.list_keys[ldx], self.list_values[ldx], self.offload_keys, self.offload_values)

        # create multi-streams and events
        self.copystream = torch.cuda.Stream()
        self.mainevents = {}
        self.copyevents = {}
        device_list = sorted(set(self.layer_mapping.values()), key=lambda x: int(x.split(':')[-1]))
        for device_idx in device_list:
            with torch.cuda.device(device_idx):
                self.mainevents[device_idx] = torch.cuda.Event()
                self.copyevents[device_idx] = torch.cuda.Event()
    
        # allocate memory for supercluster
        self.super_centroids   = []
        self.cluster_to_super  = []
        self.supercluster_size = []
        self.super_centroids_mask = []
        max_supercluster_size = 300 # this is not fixed value. segment_kmeans is not constrained to fixed cluster size.
        for ldx in range(self.layer_num):
            self.super_centroids.append(
                torch.zeros((self.batch_size*self.kv_head, self.n_super_centroids, self.head_dim), 
                            dtype=self.dtype, device=self.layer_mapping[str(ldx)]).contiguous()
            )
            self.supercluster_size.append(
                torch.zeros((self.batch_size*self.kv_head, self.n_super_centroids),
                            dtype=self.dtype, device=self.layer_mapping[str(ldx)]).contiguous()
            )
            self.cluster_to_super.append(
                torch.zeros((self.batch_size*self.kv_head, self.n_super_centroids, max_supercluster_size), 
                            dtype=self.dtype, device=self.layer_mapping[str(ldx)]).contiguous().fill_(-1)
            ) # fill -1 to represent invalid value
            self.super_centroids_mask.append(
                torch.zeros((self.batch_size*self.kv_head, self.n_super_centroids), 
                            dtype=torch.bool, device=self.layer_mapping[str(ldx)]).contiguous()
            )

        self.profile_clustering = profile_clustering

    # decide whether to pre-allocate GPU memory before prefilling
    def pre_allocate_decision(self):
        # estimate the KV Cache GPU memory consumption
        self.esitimate_gpu_memory = 2 * self.layer_num * self.batch_size * self.kv_head * (self.cache_size*self.page_size + self.n_centroids + self.static_pattern_total + self.max_new_length) * self.head_dim * 2
        self.esitimate_gpu_memory += 2 * self.batch_size * self.kv_head * self.buffer_size * self.page_size * self.head_dim * 2
        self.esitimate_gpu_memory += 2 * self.batch_size * self.kv_head * self.es_cluster_num * self.head_dim * 2
        self.esitimate_gpu_memory += 4 * self.batch_size * self.kv_head * self.group_size * self.n_centroids * 2
        self.esitimate_gpu_memory /= 1024 * 1024 * 1024
        # print(f"Estimate KV Cache GPU memory consumption: {self.esitimate_gpu_memory:.4f} GB")

        return self.free_memory > self.esitimate_gpu_memory*1.5
    
    # allocate layer-share buffer for computation
    def allocate_computation_buffer(self):
        # execution buffer to store keys & values used to compute attention, shared across layers
        self.execution_buffer_keys = torch.zeros((self.batch_size*self.kv_head, self.buffer_size*self.page_size+self.static_stride, 1, self.head_dim), 
                                                 dtype=self.dtype, device=self.layer_mapping[str(0)]).contiguous()
        self.execution_buffer_values = torch.zeros((self.batch_size*self.kv_head, self.buffer_size*self.page_size+self.static_stride, 1, self.head_dim), 
                                                   dtype=self.dtype, device=self.layer_mapping[str(0)]).contiguous()
        self.valid_lengths = torch.zeros((self.batch_size*self.kv_head), dtype=torch.int32, 
                                          device=self.layer_mapping[str(0)]).contiguous()
        self.execution_stride = self.buffer_size * self.page_size + self.static_stride
        
        # allocate layer-share buffer for batch_gemm_softmax kernel
        self.gemm_o = torch.zeros((self.batch_size, self.kv_head, self.group_size, self.n_centroids), 
                                  device=self.layer_mapping[str(0)], dtype=self.dtype).contiguous()
        self.softmax_o = torch.zeros((self.batch_size*self.kv_head, self.group_size, self.n_centroids),
                                     device=self.layer_mapping[str(0)], dtype=self.dtype).contiguous()
        self.norm = torch.zeros((self.batch_size*self.kv_head, self.group_size, (self.n_centroids+256-1)//256),
                                 device=self.layer_mapping[str(0)], dtype=torch.float32).contiguous()
        self.sum = torch.zeros((self.batch_size*self.kv_head, self.group_size, (self.n_centroids+256-1)//256),
                                device=self.layer_mapping[str(0)], dtype=torch.float32).contiguous()
        
        # allocate layer-share buffer for estimation zone
        self.es_centroids = torch.zeros((self.batch_size*self.kv_head, self.es_cluster_num, 1, self.head_dim),
                                        dtype=self.dtype, device=self.layer_mapping[str(0)]).contiguous()
        self.es_value_sum = torch.zeros((self.batch_size*self.kv_head, self.es_cluster_num, 1, self.head_dim),
                                         dtype=self.dtype, device=self.layer_mapping[str(0)]).contiguous()
        self.es_cluster_size = torch.zeros((self.batch_size*self.kv_head, 1, 1, self.es_cluster_num),
                                           dtype=self.dtype, device=self.layer_mapping[str(0)]).contiguous()

        # allocate layer-share buffer for batch_gemm_softmax kernel
        self.super_gemm_o = torch.zeros((self.batch_size, self.kv_head, self.group_size, self.n_super_centroids), 
                                  device=self.layer_mapping[str(0)], dtype=self.dtype).contiguous()
        self.super_softmax_o = torch.zeros((self.batch_size*self.kv_head, self.group_size, self.n_super_centroids),
                                     device=self.layer_mapping[str(0)], dtype=self.dtype).contiguous()
        self.super_norm = torch.zeros((self.batch_size*self.kv_head, self.group_size, (self.n_super_centroids+256-1)//256),
                                 device=self.layer_mapping[str(0)], dtype=torch.float32).contiguous()
        self.super_sum = torch.zeros((self.batch_size*self.kv_head, self.group_size, (self.n_super_centroids+256-1)//256),
                                device=self.layer_mapping[str(0)], dtype=torch.float32).contiguous()


    def prepare_cache(self):
        # sync the last batch of the last layer
        torch.cuda.synchronize()
        self.wave_buffer[self.layer_num-1].construction_sync()
        # clear temp memory
        self.clusters_cpu = None
        self.cluster_size_cpu = None
        self.temp_keys = None
        self.temp_values = None

        if not self.allocated:  # allocate GPU memory after prefilling
            self.cache_keys = []
            self.cache_values = []
            for ldx in range(self.layer_num):
                # allocate GPU Cache data
                self.cache_keys.append(
                    torch.zeros((self.batch_size, self.kv_head, self.cache_size, self.page_size, self.head_dim),
                                dtype=self.dtype, device=self.layer_mapping[str(ldx)]).contiguous()
                )
                self.cache_values.append(
                    torch.zeros((self.batch_size, self.kv_head, self.cache_size, self.page_size, self.head_dim),
                                dtype=self.dtype, device=self.layer_mapping[str(ldx)]).contiguous()
                )
                # move meta index to gpu
                self.centroids[ldx] = self.centroids[ldx].to(self.layer_mapping[str(ldx)]).contiguous()
                self.value_sum[ldx] = self.value_sum[ldx].to(self.layer_mapping[str(ldx)]).contiguous()
                self.centroids_mask[ldx] = self.centroids_mask[ldx].to(self.layer_mapping[str(ldx)]).contiguous()
                self.cluster_size[ldx] = self.cluster_size[ldx].to(self.layer_mapping[str(ldx)]).contiguous()
            self.cache_stride = self.cache_size
            self.allocate_computation_buffer()
    

    def prefill_update_kv_cache(self, query_states, key_states, value_states, layer_idx, batch_idx): 
        """
        Prefill update the key & value cache for per batch for per layer
        Args:
            query_states: [bsz, seq_len, head_num, head_dim]
            key_states: [bsz, seq_len, group_num, head_dim]
            value_states: [bsz, seq_len, group_num, head_dim]
            layer_idx: layer index
            batch_idx: batch index
        """    
        bsz, seq_len, group_num, head_dim = key_states.shape
        assert bsz == 1, f"Multi-batch prefilling only support prefill single batch one by one."
        assert seq_len <= self.input_length, f"seq_len({seq_len}) should less than input_length({self.input_length})"
        # assert group_num == self.kv_head, f"kv_head({self.kv_head}) should equal to group_num({group_num})"
        # assert head_dim == self.head_dim, f"head_dim({head_dim}) should equal to self.head_dim({self.head_dim})"

        valid_start = self.valid_start[batch_idx]
        valid_length = seq_len - self.static_pattern_total - valid_start

        # sync for the previous layer and batch finish organize pages
        if layer_idx > 0:
            self.wave_buffer[layer_idx-1].construction_sync()
        elif batch_idx > 0: # layer_idx == 0
            self.wave_buffer[self.layer_num-1].construction_sync()
        
        # store in self to avoid deleting when async offload to cpu, shape: (group_num, seq_len, dim)
        self.temp_keys = key_states[0, valid_start+self.static_pattern_start:seq_len-self.static_pattern_end, :, :].transpose(0, 1).contiguous()
        self.temp_values = value_states[0, valid_start+self.static_pattern_start:seq_len-self.static_pattern_end, :, :].transpose(0, 1).contiguous()
        self.mainevents[self.layer_mapping[str(layer_idx)]].record()

        # async offload keys & values to cpu
        with torch.cuda.stream(self.copystream):
            self.mainevents[self.layer_mapping[str(layer_idx)]].wait()
            self.offload_keys[:, :valid_length, :].copy_(self.temp_keys, non_blocking=True)
            self.offload_values[:, :valid_length, :].copy_(self.temp_values, non_blocking=True)
            self.copyevents[self.layer_mapping[str(layer_idx)]].record()
        
        # copy steady zone to pre-allocated memory
        self.steady_zone_keys[layer_idx][batch_idx, :, :self.static_pattern_start, :] = \
            key_states[0, valid_start:valid_start+self.static_pattern_start, :, :].transpose(0, 1)
        self.steady_zone_keys[layer_idx][batch_idx, :, self.static_pattern_start:self.static_pattern_total, :] = \
            key_states[0, seq_len-self.static_pattern_end:seq_len, :, :].transpose(0, 1)
        self.steady_zone_values[layer_idx][batch_idx, :, :self.static_pattern_start, :] = \
            value_states[0, valid_start:valid_start+self.static_pattern_start, :, :].transpose(0, 1)
        self.steady_zone_values[layer_idx][batch_idx, :, self.static_pattern_start:self.static_pattern_total, :] = \
            value_states[0, seq_len-self.static_pattern_end:seq_len, :, :].transpose(0, 1)

        # compute key mean, shape (group_num, 1, head_dim)
        mean_key = torch.mean(self.temp_keys, dim=1, keepdim=True)

        # segmented k-means
        # import time
        # start_time = time.time()
        # torch.cuda.synchronize()
        _centroids, _value_sum, _clusters, _cluster_size = segment_k_means(
            key=self.temp_keys-mean_key,    # centering to 0
            value=self.temp_values,
            num_centroids=self.n_centroids,
            num_segments=self.n_segment,
        )

        # compute key mean, shape (group_num, 1, head_dim)
        mean_centroid = torch.mean(_centroids, dim=1, keepdim=True)

        # cluster those centroids into “superclusters”
        _supercentroids, _, _superclusters, _supercluster_size = segment_k_means(
            key = _centroids-mean_centroid,
            value = None,
            num_centroids=self.n_super_centroids,
            num_iters = 100,
            num_segments=1,
        )
        # super_centroids: [S×D]
        # cluster_to_super: [K]   maps each of the K clusters → a supercluster id in [0..S-1]
        self.super_centroids[layer_idx]   = _supercentroids + mean_centroid
        self.cluster_to_super[layer_idx]  = _superclusters
        self.supercluster_size[layer_idx] = _supercluster_size
        self.super_centroids_mask[layer_idx] = (_supercluster_size == 0)          # (group_num, n_super_centroids)
        
        # torch.cuda.synchronize()
        # end_time = time.time()
        # print(f"prefill clustering:{end_time - start_time}")
        # assert _centroids.shape[-2] == _value_sum.shape[-2] == _cluster_size.shape[-1] == _clusters.shape[-2] == self.n_centroids

        # copy meta index
        self.centroids[layer_idx][batch_idx*self.kv_head:(batch_idx+1)*self.kv_head, :, :].copy_(_centroids + mean_key)         # (group_num, n_centroids, dim)
        self.value_sum[layer_idx][batch_idx*self.kv_head:(batch_idx+1)*self.kv_head, :, :].copy_(_value_sum)                    # (group_num, n_centroids, dim)
        self.centroids_mask[layer_idx][batch_idx*self.kv_head:(batch_idx+1)*self.kv_head, :].copy_(_cluster_size == 0)          # (group_num, n_centroids)
        self.cluster_size[layer_idx][batch_idx*self.kv_head:(batch_idx+1)*self.kv_head, :].copy_(_cluster_size.to(self.dtype))  # (group_num, n_centroids)

        # these data will be used to organize the cpu kv
        self.cluster_size_cpu = _cluster_size.cpu().contiguous()    # (group_num, n_centroids)
        self.clusters_cpu = _clusters.cpu().contiguous()            # (group_num, n_centroids, max_cluster_size)
        
        if (layer_idx == self.layer_num - 1) and (batch_idx + bsz == self.batch_size):
            self.context += seq_len

        # save cluster information for simulation
        if self.profile_clustering:
            import os
            self.outdir_path = f"/home/juchanlee/MagicDec/Engine/RetrievalAttention/profile/data/data_superclustersize_{self.approx_supercluster_size}_0.125KV"
            os.makedirs(self.outdir_path, exist_ok=True)
            torch.save(_centroids, f"{self.outdir_path}/centroid_{layer_idx}.pt")
            torch.save(_cluster_size, f"{self.outdir_path}/cluster_size_{layer_idx}.pt")
            torch.save(_clusters, f"{self.outdir_path}/clusters_{layer_idx}.pt")
            torch.save(_supercentroids, f"{self.outdir_path}/supercentroids_{layer_idx}.pt")
            torch.save(_supercluster_size, f"{self.outdir_path}/supercluster_size_{layer_idx}.pt")
            torch.save(_superclusters, f"{self.outdir_path}/superclusters_{layer_idx}.pt")
                
        return key_states[:, valid_start:, :, :], value_states[:, valid_start:, :, :]   # ignore mask tokens, shape: (bsz, seq_len, group_num, dim)

    def sync(
        self,
        layer_idx,
        batch_idx
    ):  
        """
        wait async offloading on copystream -> organize kv
        """
        # wait for offload finish
        self.copyevents[self.layer_mapping[str(layer_idx)]].synchronize()
        # async organize kv
        self.wave_buffer[layer_idx].async_construction(
            self.clusters_cpu,      # (group_num, n_centroids, max_cluster_size)
            self.cluster_size_cpu,  # (group_num, n_centroids)
            batch_idx
        )


    # update KV cache when generate tokens exceed THRESHOLD_LENGTH
    def _update_kv_cache(self):
        for ldx in range(self.layer_num):
            update_keys = self.steady_zone_keys[ldx][:, :, self.static_pattern_start:self.static_pattern_total-self.static_pattern_end, :].clone().reshape(self.batch_size*self.kv_head, THRESHOLD_LENGTH, self.head_dim).contiguous()
            update_values = self.steady_zone_values[ldx][:, :, self.static_pattern_start:self.static_pattern_total-self.static_pattern_end, :].clone().reshape(self.batch_size*self.kv_head, THRESHOLD_LENGTH, self.head_dim).contiguous()
            self.mainevents[self.layer_mapping[str(ldx)]].record()

            # move local window
            self.steady_zone_keys[ldx][:, :, self.static_pattern_start:self.static_pattern_start+self.static_pattern_end, :] = \
                self.steady_zone_keys[ldx][:, :, self.static_pattern_total-self.static_pattern_end:self.static_pattern_total, :]
            self.steady_zone_values[ldx][:, :, self.static_pattern_start:self.static_pattern_start+self.static_pattern_end, :] = \
                self.steady_zone_values[ldx][:, :, self.static_pattern_total-self.static_pattern_end:self.static_pattern_total, :]

            # async offload
            with torch.cuda.stream(self.copystream):
                self.mainevents[self.layer_mapping[str(ldx)]].wait()
                self.offload_update_keys.copy_(update_keys, non_blocking=True)
                self.offload_update_values.copy_(update_values, non_blocking=True)
                self.copyevents[self.layer_mapping[str(ldx)]].record()
            
            # compute key mean, shape (batch_size*group_num, 1, head_dim)
            mean_key = torch.mean(update_keys, dim=1, keepdim=True)
            
            # segmented k-means
            # import time
            # start_time = time.time()
            # torch.cuda.synchronize()            
            _centroids, _value_sum, _clusters, _cluster_size = segment_k_means(
                key=update_keys-mean_key,   # centering to 0, (batch_size*group_num, THRESHOLD_LENGTH, dim)
                value=update_values,        # (batch_size*group_num, THRESHOLD_LENGTH, dim)
                num_centroids=self.n_centroids_per_update_segment,
                num_segments=1,
            )
            # torch.cuda.synchronize()
            # end_time = time.time()
            # print(f"decode clustering:{end_time - start_time}")

            _centroids += mean_key
            assert _centroids.shape[-2] == _value_sum.shape[-2] == _cluster_size.shape[-1] == _clusters.shape[-2] == self.n_centroids_per_update_segment

            # append to meta index
            self.centroids[ldx] = torch.cat((self.centroids[ldx], _centroids), dim=1)  # (batch_szie*group_num, new_n_centroids, dim)
            self.value_sum[ldx] = torch.cat((self.value_sum[ldx], _value_sum), dim=1)  # (batch_szie*group_num, new_n_centroids, dim)
            self.centroids_mask[ldx] = torch.cat((self.centroids_mask[ldx], _cluster_size == 0), dim=1) # (batch_szie*group_num, new_n_centroids)
            self.cluster_size[ldx] = torch.cat((self.cluster_size[ldx], _cluster_size.to(self.dtype)), dim=1) # (batch_szie*group_num, new_n_centroids)
            assert self.centroids[ldx].shape[-2] == self.value_sum[ldx].shape[-2] == self.centroids_mask[ldx].shape[-1] == self.cluster_size[ldx].shape[-1] == self.n_centroids + self.n_centroids_per_update_segment

            # update wave buffer
            self.copyevents[self.layer_mapping[str(ldx)]].synchronize()
            self.wave_buffer[ldx].update_kv(
                self.offload_update_keys,           # (batch_size*group_num, THRESHOLD_LENGTH, dim)
                self.offload_update_values,         # (batch_size*group_num, THRESHOLD_LENGTH, dim)
                _clusters.cpu().contiguous(),       # (batch_size*group_num, n_centroids_per_update_segment, max_cluster_size)
                _cluster_size.cpu().contiguous()    # (batch_size*group_num, n_centroids_per_update_segment)
            )
        
        # update n_centroids
        self.n_centroids += self.n_centroids_per_update_segment
        # re-allocate layer-share buffer for batch_gemm_softmax kernel
        self.gemm_o = torch.zeros((self.batch_size, self.kv_head, self.group_size, self.n_centroids), 
                                  device=self.layer_mapping[str(0)], dtype=self.dtype).contiguous()
        self.softmax_o = torch.zeros((self.batch_size*self.kv_head, self.group_size, self.n_centroids),
                                     device=self.layer_mapping[str(0)], dtype=self.dtype).contiguous()
        self.norm = torch.zeros((self.batch_size*self.kv_head, self.group_size, (self.n_centroids+256-1)//256),
                                 device=self.layer_mapping[str(0)], dtype=torch.float32).contiguous()
        self.sum = torch.zeros((self.batch_size*self.kv_head, self.group_size, (self.n_centroids+256-1)//256),
                                device=self.layer_mapping[str(0)], dtype=torch.float32).contiguous()
        # reset static pattern
        self.static_pattern_total = self.static_pattern_start + self.static_pattern_end


    def decode_update_kv_cache(self,
        key_states,         # (bs, length(=1), group_num, dim)
        value_states,       # (bs, length(=1), group_num, dim)
        layer_idx
    ):
        # index update
        if self.static_pattern_total == self.static_pattern_start + self.static_pattern_end + THRESHOLD_LENGTH:
            # print("Updating KV cache ...")
            self._update_kv_cache()
            # print("KV cache updated, continue decoding ...")

        # append newly generated token to the steady zone
        self.steady_zone_keys[layer_idx][:, :, self.static_pattern_total, :] = key_states[:, 0, :, :]
        self.steady_zone_values[layer_idx][:, :, self.static_pattern_total, :] = value_states[:, 0, :, :]

        if layer_idx == self.layer_num - 1:
            self.context += 1
            self.static_pattern_total += 1

        return None, None   # no use the return value
    

    def compute(self, queries, layer_idx):
        """
        queries: query vector, shape: (batch_size, 1, head_num, dim), gpu torch tensor
        """
        # assert queries.size(0) == self.batch_size
        # assert queries.size(1) == 1
        # assert queries.size(2) == self.kv_head * self.group_size == self.num_heads
        # assert queries.size(3) == self.head_dim

        static_len = self.static_pattern_total if layer_idx == self.layer_num - 1 else self.static_pattern_total + 1

        # import time
        # start_time = time.time()
        # torch.cuda.synchronize()

        # search for TopK centroids
        batch_gemm_softmax(queries, self.centroids[layer_idx], self.gemm_o, self.norm, self.sum, self.softmax_o,
                           self.batch_groups, self.group_size, self.n_centroids, self.head_dim,
                           self.RSQRT_DIM, 0)       # [batch_size*group_num, group_size, n_centroids]
        dist = torch.sum(self.softmax_o, dim=1)     # [batch_size*group_num, n_centroids]
        dist.masked_fill_(self.centroids_mask[layer_idx], self.DTYPE_MIN)
        cI = torch.topk(dist, self.max_compute_cluster_num, dim=-1, largest=True, sorted=True)[1] # [batch_size*group_num, max_consider_cluster]

        if self.profile_clustering:
            # compare with supercluster entry and only select top
            # search for TopK supercentroids
            batch_gemm_softmax(queries, self.super_centroids[layer_idx], self.super_gemm_o, self.super_norm, self.super_sum, self.super_softmax_o,
                            self.batch_groups, self.group_size, self.n_super_centroids, self.head_dim,
                            self.RSQRT_DIM, 0)       # [batch_size*group_num, group_size, n_centroids]
            super_dist = torch.sum(self.super_softmax_o, dim=1)     # [batch_size*group_num, n_centroids]
            super_dist.masked_fill_(self.super_centroids_mask[layer_idx], self.DTYPE_MIN)
            super_cI = torch.topk(super_dist, int(self.max_compute_cluster_num/self.approx_supercluster_size), dim=-1, largest=True, sorted=True)[1] # [batch_size*group_num, max_consider_cluster]
            
            # select only clusters from selected supercluster and see what happens
            cI_of_selected_superclusters = torch.zeros((super_cI.shape[0], self.max_compute_cluster_num), dtype=torch.int32, device=self.layer_mapping[str(layer_idx)])
            for head_idx in range(super_cI.shape[0]):
                selected_cluster_counter_per_head = 0
                for super_cid in super_cI[head_idx]:
                    supercluster_size = self.supercluster_size[layer_idx][head_idx][super_cid]
                    
                    start_idx = selected_cluster_counter_per_head
                    end_idx = selected_cluster_counter_per_head + supercluster_size

                    # sanity check for indexing error
                    if end_idx > self.max_compute_cluster_num:
                        break
                    
                    cI_of_selected_superclusters[head_idx, start_idx:end_idx] = self.cluster_to_super[layer_idx][head_idx][super_cid][:supercluster_size]
                    
                    selected_cluster_counter_per_head += supercluster_size

        # # with super_cI, get selectd supercluster's clusters.
        selection_method = "fine-grained"
        if selection_method == "fine-grained":
          self.cluster_ids.copy_(cI[..., :self.nprobe])
        elif selection_method == "coarse-grained":
          self.cluster_ids.copy_(cI_of_selected_superclusters[..., :self.nprobe])


        # torch.cuda.synchronize()
        # end_time = time.time()
        # print(f"cluster selection:{end_time - start_time}")

        if self.profile_clustering:
            # calculate how many cluster are included in topk(self.nprobe)
            selected_cluster_num = torch.zeros((self.batch_size*self.kv_head, self.n_super_centroids))
            selected_cluster_per_supercluster = []
            selected_cI = cI[..., :self.nprobe]
            cI_of_selected_superclusters = cI_of_selected_superclusters[..., :self.nprobe]
            for batch_idx in range(self.batch_size*self.kv_head):
                selected_cluster_per_supercluster_tmp = [[] for _ in range(self.n_super_centroids)]

                for supercluster_idx in range(self.n_super_centroids):
                    for cluster_idx in range(self.cluster_to_super[layer_idx].shape[-1]):
                        if self.cluster_to_super[layer_idx][batch_idx][supercluster_idx][cluster_idx] in selected_cI[batch_idx]:
                            selected_cluster_num[batch_idx][supercluster_idx] += 1
                            selected_cluster_per_supercluster_tmp[supercluster_idx].append(self.cluster_to_super[layer_idx][batch_idx][supercluster_idx][cluster_idx])

                selected_cluster_per_supercluster.append(selected_cluster_per_supercluster_tmp)

            selected_cluster_ratio = torch.zeros((self.batch_size*self.kv_head, self.n_super_centroids))
            # calculate selection ratio of each supercluster
            for batch_idx in range(self.batch_size*self.kv_head):
                for supercluster_idx in range(self.n_super_centroids):
                    selected_cluster_ratio[batch_idx][supercluster_idx] =  selected_cluster_num[batch_idx][supercluster_idx] / self.supercluster_size[layer_idx][batch_idx][supercluster_idx] * 100

            # # save cluster information for simulation
            torch.save(selected_cI, f"{self.outdir_path}/selected_cI_{layer_idx}.pt")
            torch.save(selected_cluster_num, f"{self.outdir_path}/selected_cluster_num_{layer_idx}.pt")
            torch.save(selected_cluster_ratio, f"{self.outdir_path}/selected_cluster_ratio_{layer_idx}.pt")
            torch.save(selected_cluster_per_supercluster, f"{self.outdir_path}/selected_cluster_per_supercluster_{layer_idx}.pt")
            torch.save(selected_cluster_per_supercluster, f"{self.outdir_path}/selected_cluster_per_supercluster_{layer_idx}.pt")
            torch.save(cI_of_selected_superclusters, f"{self.outdir_path}/cI_of_selected_superclusters_{layer_idx}.pt")

            import matplotlib.pyplot as plt
            import numpy as np

            # Assume `selected_cluster_ratio` is a torch.Tensor of shape [batch_size * kv_head, n_super_centroids]
            # Flatten to 1D numpy array of percentages
            ratios = selected_cluster_ratio.cpu().flatten().numpy()

            # Define bins for 0-100% in 10% increments
            bins = np.arange(0, 110, 10)

            plt.figure()
            plt.hist(ratios, bins=bins, edgecolor='black')
            plt.xlabel('Selected Cluster Ratio (%)')
            plt.ylabel('Count')
            plt.title('Selected Cluster Ratios per Supercluster')
            plt.xticks(bins)
            # Save the figure to a PNG file
            output_path = '/home/juchanlee/MagicDec/selected_cluster_ratio_hist_4supercluster_100iter.png'
            plt.savefig(output_path)
            plt.close()
            plt.show()

            nz_ratios = ratios[ratios > 0]  # Filter out zero values
            plt.figure()
            plt.hist(nz_ratios, bins=bins, edgecolor='black')
            plt.xlabel('Selected Cluster Ratio (%)')
            plt.ylabel('Count')
            plt.title('Histogram of Selected Cluster Ratios')
            plt.xticks(bins)
            # Save the figure to a PNG file
            output_path = '/home/juchanlee/MagicDec/selected_cluster_ratio_hist_wo_zero_4supercluster_100iter.png'
            plt.savefig(output_path)
            plt.close()
            plt.show()

        # estimation zone computation
        if self.es_cluster_num > 0:
            gather_copy_vectors(self.centroids[layer_idx], self.es_centroids, 
                                self.value_sum[layer_idx], self.es_value_sum, 
                                self.cluster_size[layer_idx], self.es_cluster_size,
                                cI, self.batch_groups, self.n_centroids, self.es_cluster_num, 
                                self.max_compute_cluster_num, self.nprobe, self.es_cluster_num)
            
            es_out, es_lse = weighted_flash_decoding(
                queries.view(self.batch_groups, 1, self.group_size, self.head_dim), 
                self.es_centroids,       # [batch_size*group_num, es_cluster, 1, dim]
                self.es_value_sum,       # [batch_size*group_num, es_cluster, 1, dim]
                self.es_cluster_size,    # [batch_size*group_num, 1, 1, es_cluster]
                previous_out=None, previous_lse=None,
                return_softmax_lse=True)
        else:
            es_out, es_lse = None, None
        
        # cache access and submit cache update tasks to thread pool
        self.wave_buffer[layer_idx].batch_access()

        # assemble the execution buffer
        gather_copy_and_concat(self.steady_zone_keys[layer_idx], self.list_keys[layer_idx], self.cache_keys[layer_idx], self.execution_buffer_keys, 
                               self.steady_zone_values[layer_idx], self.list_values[layer_idx], self.cache_values[layer_idx], self.execution_buffer_values,
                               self.miss_unit_idices[layer_idx], self.miss_unit_sizes[layer_idx], self.miss_unit_sizes_cumsum[layer_idx], self.miss_num_units[layer_idx],
                               self.hit_unit_idices[layer_idx], self.hit_unit_sizes[layer_idx], self.hit_unit_sizes_cumsum[layer_idx], self.hit_num_units[layer_idx],
                               self.valid_lengths, self.batch_groups, 
                               self.static_stride, self.list_stride, self.cache_stride,
                               self.execution_stride, self.buffer_size, static_len)

        # flash attention for retrieve zone and steady zone, merge the estimation zone results at the same time
        attn_out = weighted_flash_decoding(
            queries.view(self.batch_groups, 1, self.group_size, self.head_dim), 
            self.execution_buffer_keys,    # (batch_size*group_num, execution_stride, 1, dim)
            self.execution_buffer_values,  # (batch_size*group_num, execution_stride, 1, dim)
            previous_out=es_out,
            previous_lse=es_lse,
            cache_seqlens=self.valid_lengths,
            return_softmax_lse=False
        )

        # admiss pages from execution buffer to GPU cache
        self.wave_buffer[layer_idx].sync()  # wait for update LRU finish
        gather_copy_and_scatter(self.execution_buffer_keys, self.cache_keys[layer_idx], self.execution_buffer_values, self.cache_values[layer_idx],
                                self.update_buffer_indices[layer_idx], self.update_unit_sizes[layer_idx], self.update_cache_indices[layer_idx], 
                                self.update_num_units[layer_idx], self.batch_groups, self.execution_stride, self.cache_stride,
                                self.buffer_size, static_len)

        return attn_out.view(self.batch_size, 1, self.num_heads, self.head_dim)
