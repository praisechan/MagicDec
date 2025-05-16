
#ifndef COPY_CUH
#define COPY_CUH

// chunk_size = 8, dim = 128, float16 ==> 8*128*2=2048 Bytes
// PTYPE = int2 (8 Bytes) ==> BLOCK_SIZE_CP = 256 (one chunk has 256 int2 data)
#define CHUNK_DATA_SIZE (CHUNK_SIZE * 128 * DATA_BYTES)
#define BLOCK_SIZE_CP (CHUNK_DATA_SIZE / sizeof(PTYPE))

// one vector has (128*2/8)=32 int2 data
#define VECTOR_SIZE_CP ((128 * DATA_BYTES) / sizeof(PTYPE))

#define SPLIT_FACTOR 16


// copy the vector from src based on src_index to dst
template <typename T>
__device__ __forceinline__ void random_vector_copy_kernel(
    T *src,
    T *dst,
    int* src_index,         // src vector-level index
    int src_size_per_group, // data number per group for src
    int dst_size_per_group, // data number per group for dst
    int cpy_vector_num,     // copy vector number
    int dst_start,          // dst start index (vector-level)
    int bid)                // group id
{
    int64_t bid_64 = bid; // need to cast to int64_t to make sure index not overflow.
    int64_t src_base = (bid_64 * src_size_per_group) * DATA_BYTES / sizeof(T);
    int64_t dst_base = (bid_64 * dst_size_per_group) * DATA_BYTES / sizeof(T); 
    dst_base += dst_start * VECTOR_SIZE_CP;
    int copy_size = cpy_vector_num * VECTOR_SIZE_CP;   // copy size in sizeof(T) Bytes

#pragma unroll
    for (int idx = threadIdx.x; idx < copy_size; idx += BLOCK_SIZE_CP) {
        int src_idx = src_index[idx / VECTOR_SIZE_CP];
        dst[dst_base + idx] = src[src_base + src_idx * VECTOR_SIZE_CP + idx % VECTOR_SIZE_CP];
    }
}

// copy vector from src to dst (for both keys, values and metadata)
// the sre indices are randomly, but copied to contiguous dst
template <typename T, typename M> 
__global__ void random_gather_copy_vector(
    T *src_keys, T *dst_keys,           // src and dst for keys
    T *src_values, T *dst_values,       // src and dst for values
    M* src_metadata, M* dst_metadata,   // src and dst for metadata

    int src_size_per_group,     // vector number per group for src
    int dst_size_per_group,     // vector number per group for dst

    int64_t *src_indices,       // src vector-level index, shape (group_num, index_size)
    int index_size,             // src_indices size per group
    int index_start,            // valid start index for src_indices
    int src_cpy_size            // total number of vectors need to be copy from src
) {
    extern __shared__ int s[];
    int* src_idx = s;    // vector-level start index of src for each selected clusters
    
    // SPLIT_FACTOR blocks handle one group copy
    int bid = blockIdx.x / SPLIT_FACTOR;
    int split_id = blockIdx.x % SPLIT_FACTOR;

    int split_size = (src_cpy_size + SPLIT_FACTOR - 1) / SPLIT_FACTOR;
    int start = split_id * split_size;
    int number = min(split_size, src_cpy_size - start);

    // copy indices to shared memory
    int index_offset = bid * index_size + index_start + start;
#pragma unroll
    for (int idx = threadIdx.x; idx < number; idx += BLOCK_SIZE_CP) {
        src_idx[idx] = static_cast<int>(src_indices[index_offset + idx]);
    }
    __syncthreads();

    // copy data
    random_vector_copy_kernel(src_keys, dst_keys, src_idx, src_size_per_group*128, dst_size_per_group*128, number, start, bid);
    random_vector_copy_kernel(src_values, dst_values, src_idx, src_size_per_group*128, dst_size_per_group*128, number, start, bid);

    // copy metadata
    int64_t bid_64 = bid; // need to cast to int64_t to make sure index not overflow.
    int64_t src_base = (bid_64 * src_size_per_group);
    int64_t dst_base = (bid_64 * dst_size_per_group) + start;
#pragma unroll
    for (int idx = threadIdx.x; idx < number; idx += BLOCK_SIZE_CP) {
        dst_metadata[dst_base + idx] = src_metadata[src_base + src_idx[idx]];
    }
}



// A chunk-level gather-copy kernel
// src_index is chunk-level index, vectors in each chunk are not always fully copy
// src_cpy_size is the number of vectors need to be copied for each chunk
// dst_index is vector-level index for each chunk start position, which is the same as cumulative sum of src_cpy_size
template <typename T>
__device__ __forceinline__ void chunk_gather_copy_kernel(
    T *src,
    T *dst,
    int *src_index,         // src chunk-level index
    int *src_cpy_size,      // number of vectors need to be copy for each chunk
    int *dst_index,         // dst vector-level index
    int *size_thread_map,   // map from copy vector number to maximum thread number
    int src_size_per_group, // data number per group for src
    int dst_size_per_group, // data number per group for dst
    int dst_start,          // dst start index (vector-level)
    int cpy_chunk_num,      // chunk num need to be copy
    int bid)                // group id
{
    // one CUDA Block (BLOCK_SIZE_CP threads) handles one chunk data each time,
    // each thread copies sizeof(T) Bytes data once,
    // loop over src_index to copy multiple chunks data.
    // if some of the chunk data is no need to copy, threads for those positions just skip them.

    int64_t bid_64 = bid; // need to cast to int64_t to make sure index not overflow.
    int64_t src_base = (bid_64 * src_size_per_group) * DATA_BYTES / sizeof(T);
    int64_t dst_base = (bid_64 * dst_size_per_group) * DATA_BYTES / sizeof(T); 
    dst_base += dst_start * VECTOR_SIZE_CP;

    int offset_idx = -1;

    // copy chunk by chunk
#pragma unroll
    while (offset_idx < cpy_chunk_num - 1)
    {
        offset_idx++;
        
        int cpy_size = src_cpy_size[offset_idx];        // number of vectors need to be copy for this chunk
        if (threadIdx.x >= size_thread_map[cpy_size])   // skip this thread for this chunk
            continue;
                
        // compute src and dst offset
        int64_t src_offset = src_base + src_index[offset_idx] * BLOCK_SIZE_CP + threadIdx.x;
        int64_t dst_offset = dst_base + dst_index[offset_idx] * VECTOR_SIZE_CP + threadIdx.x;
        // copy data
        dst[dst_offset] = src[src_offset];
    }
}

// A chunk-level gather-copy kernel with vector-level index
// src_index is vector-level index, indicates the start vector index of each chunk
// src_cpy_size is the number of vectors need to be copied for each chunk
// dst_index is vector-level index for each chunk start position, which is the same as cumulative sum of src_cpy_size
template <typename T>
__device__ __forceinline__ void chunk_gather_copy_with_vector_indices_kernel(
    T *src,
    T *dst,
    int *src_index,         // src vector-level index
    int *src_cpy_size,      // number of vectors need to be copy for each chunk
    int *dst_index,         // dst vector-level index
    int *size_thread_map,   // map from copy vector number to maximum thread number
    int src_size_per_group, // data number per group for src
    int dst_size_per_group, // data number per group for dst
    int dst_start,          // dst start index (vector-level)
    int cpy_chunk_num,      // chunk num need to be copy
    int bid)                // group id
{
    // one CUDA Block (BLOCK_SIZE_CP threads) handles one chunk data each time,
    // each thread copies sizeof(T) Bytes data once,
    // loop over src_index to copy multiple chunks data.
    // if some of the chunk data is no need to copy, threads for those positions just skip them.

    int64_t bid_64 = bid; // need to cast to int64_t to make sure index not overflow.
    int64_t src_base = (bid_64 * src_size_per_group) * DATA_BYTES / sizeof(T);
    int64_t dst_base = (bid_64 * dst_size_per_group) * DATA_BYTES / sizeof(T); 
    dst_base += dst_start * VECTOR_SIZE_CP;

    int offset_idx = -1;

    // copy chunk by chunk
#pragma unroll
    while (offset_idx < cpy_chunk_num - 1)
    {
        offset_idx++;
        
        int cpy_size = src_cpy_size[offset_idx];        // number of vectors need to be copy for this chunk
        if (threadIdx.x >= size_thread_map[cpy_size])   // skip this thread for this chunk
            continue;
                
        // compute src and dst offset
        int64_t src_offset = src_base + src_index[offset_idx] * VECTOR_SIZE_CP + threadIdx.x;
        int64_t dst_offset = dst_base + dst_index[offset_idx] * VECTOR_SIZE_CP + threadIdx.x;
        // copy data
        dst[dst_offset] = src[src_offset];
    }
}

// A vector-level copy kernel
// copy the first `cpy_vector_num` vector from src to dst
template <typename T>
__device__ __forceinline__ void vector_copy_kernel(
    T *src,
    T *dst,
    int src_size_per_group, // data number per group for src
    int dst_size_per_group, // data number per group for dst
    int cpy_vector_num,     // copy vector number
    int bid)                // group id
{
    int64_t bid_64 = bid; // need to cast to int64_t to make sure index not overflow.
    int64_t src_base = (bid_64 * src_size_per_group) * DATA_BYTES / sizeof(T);
    int64_t dst_base = (bid_64 * dst_size_per_group) * DATA_BYTES / sizeof(T); 
    int copy_size = cpy_vector_num * VECTOR_SIZE_CP;   // copy size in sizeof(T) Bytes

#pragma unroll
    for (int idx = threadIdx.x; idx < copy_size; idx += BLOCK_SIZE_CP) {
        dst[dst_base + idx] = src[src_base + idx];
    }
}

// gather copy from three src and concat them into one buffer (for both keys and values)
// each group uses SPLIT_FACTOR CUDA blocks to handle copy
// the first block copy src1 (vector-level)
// the rest blocks copy src2 and src3 (chunk-level)
// blocks used to copy from src2 and src3 are dynamically allocated based on copy size
template <typename T> 
__global__ void concat_gather_copy(
    T *src_keys1, T *src_keys2, T *src_keys3, T *dst_keys,          // three src and one dst for keys
    T *src_values1, T *src_values2, T *src_values3, T *dst_values,  // three src and one dst for values

    int src1_size_per_group,    // data number per group for src1
    int src2_size_per_group,    // data number per group for src2
    int src3_size_per_group,    // data number per group for src3
    int dst_size_per_group,     // data number per group for dst

    int src1_cpy_num,           // number of vectors need to be copy from src1

    int index_size,             // index buffer size per group for the following index buffers

    int *src2_index,            // src2 start vector-level index for each chunk, shape (group_num, index_size)
    int *src2_cpy_size,         // number of vectors need to be copy for each chunk in src2, shape (group_num, index_size)
    int *dst_index2,            // dst vector-level index for src2, shape (group_num, index_size)
    int *src2_cpy_num,          // number of chunks need to be copy for each group in src2, shape (group_num)

    int *src3_index,            // src3 chunk-level index, shape (group_num, index_size)
    int *src3_cpy_size,         // number of vectors need to be copy for each chunk in src3, shape (group_num, index_size)
    int *dst_index3,            // dst vector-level index for src3, shape (group_num, index_size)
    int *src3_cpy_num,          // number of chunks need to be copy for each group in src3, shape (group_num)

    int *valid_vector_num       // output, valid vector number for each group, shape (group_num)
) {
    extern __shared__ int s[];  // shared_memory pointers

    // SPLIT_FACTOR blocks handle one group copy
    int bid = blockIdx.x / SPLIT_FACTOR;
    int split_id = blockIdx.x % SPLIT_FACTOR;

    int src2_cpy_chunk_num = src2_cpy_num[bid];     // number of chunks need to be copy from src2
    int src3_cpy_chunk_num = src3_cpy_num[bid];     // number of chunks need to be copy from src3

    // the first block copy data from src1
    if (split_id == 0) {
        vector_copy_kernel(src_keys1, dst_keys, src1_size_per_group, dst_size_per_group, src1_cpy_num, bid);
        vector_copy_kernel(src_values1, dst_values, src1_size_per_group, dst_size_per_group, src1_cpy_num, bid);

        // number of vectors need to be copy from src2
        int src2_cpy_vector_num = src2_cpy_chunk_num == 0 ? 0 : (dst_index2[bid * index_size + src2_cpy_chunk_num - 1] + 
                                                                src2_cpy_size[bid * index_size + src2_cpy_chunk_num - 1]);
        // number of vectors need to be copy from src3
        int src3_cpy_vector_num = src3_cpy_chunk_num == 0 ? 0 : (dst_index3[bid * index_size + src3_cpy_chunk_num - 1] + 
                                                                src3_cpy_size[bid * index_size + src3_cpy_chunk_num - 1]);
        // total number of vectors need to be copy for this group
        valid_vector_num[bid] = src1_cpy_num + src2_cpy_vector_num + src3_cpy_vector_num;
    
    // the rest blocks copy data from src2 and src3
    } else {
        int* size_thread_map = s;           // map from copy vector number to maximum thread number, 0~CHUNK_SIZE
        if (threadIdx.x <= CHUNK_SIZE) {    // we assume CHUNK_SIZE <= BLOCK_SIZE_CP
            size_thread_map[threadIdx.x] = threadIdx.x * VECTOR_SIZE_CP;    // copy i vectors need i * VECTOR_SIZE_CP threads
        }
        int* src_index = s + CHUNK_SIZE + 1;            // shared memory for src chunk-level index
        int* src_cpy_size = src_index + index_size;     // shared memory for src copy vector number for each chunk
        int* dst_index = src_cpy_size + index_size;     // shared memory for dst vector-level index

        split_id -= 1;  // split_id start from 1
        // dynamic allocate CUDA blocks for copy from src2 and src3
        int total_cpy_chunk_num = src2_cpy_chunk_num + src3_cpy_chunk_num;
        int TOTAL_BLOCKS = SPLIT_FACTOR - 1;
        // number of CUDA blocks used to copy from src2
        int BLOCKS_2 = min((TOTAL_BLOCKS * src2_cpy_chunk_num + total_cpy_chunk_num - 1) / total_cpy_chunk_num, 
                           TOTAL_BLOCKS - 1);   // when cpy2 >> cpy3, but cp3 > 0, BLOCKS_2 = TOTAL_BLOCKS - 1
        if (src3_cpy_chunk_num == 0) BLOCKS_2 = TOTAL_BLOCKS;  // however, when cpy3 == 0, BLOCKS_2 = TOTAL_BLOCKS
        // number of CUDA blocks used to copy from src3
        int BLOCKS_3 = TOTAL_BLOCKS - BLOCKS_2;

        // if (threadIdx.x == 0 && split_id == 0) {
        //     printf("group %d, src2_cpy_chunks = %d, blocks_for_src2 = %d, src3_cpy_chunks = %d, blocks_for_src3 = %d\n", bid, src2_cpy_chunk_num, BLOCKS_2, src3_cpy_chunk_num, BLOCKS_3);
        // }

        if (split_id < BLOCKS_2) {  // copy from src2
            int split_chunk = (src2_cpy_chunk_num + BLOCKS_2 - 1) / BLOCKS_2;
            int start = split_id * split_chunk;
            int number = min(split_chunk, src2_cpy_chunk_num - start);

            // copy indices to shared memory
            int index_offset = bid * index_size + start;
            for (int idx = threadIdx.x; idx < number; idx += BLOCK_SIZE_CP) {
                src_index[idx] = src2_index[index_offset + idx];
                src_cpy_size[idx] = src2_cpy_size[index_offset + idx];
                dst_index[idx] = dst_index2[index_offset + idx];
            }
            __syncthreads();

            chunk_gather_copy_with_vector_indices_kernel(src_keys2, dst_keys, src_index, src_cpy_size, dst_index, size_thread_map, src2_size_per_group, dst_size_per_group, src1_cpy_num, number, bid);
            chunk_gather_copy_with_vector_indices_kernel(src_values2, dst_values, src_index, src_cpy_size, dst_index, size_thread_map, src2_size_per_group, dst_size_per_group, src1_cpy_num, number, bid);

        } else {    // copy from src3
            split_id -= BLOCKS_2;
            int split_chunk = (src3_cpy_chunk_num + BLOCKS_3 - 1) / BLOCKS_3;
            int start = split_id * split_chunk;
            int number = min(split_chunk, src3_cpy_chunk_num - start);
            
            // copy indices to shared memory
            int index_offset = bid * index_size + start;
            for (int idx = threadIdx.x; idx < number; idx += BLOCK_SIZE_CP) {
                src_index[idx] = src3_index[index_offset + idx];
                src_cpy_size[idx] = src3_cpy_size[index_offset + idx];
                dst_index[idx] = dst_index3[index_offset + idx];
            }
            __syncthreads();

            // number of vectors need to be copy from src2
            int src2_cpy_vector_num = src2_cpy_chunk_num == 0 ? 0 : (dst_index2[bid * index_size + src2_cpy_chunk_num - 1] + 
                                                                    src2_cpy_size[bid * index_size + src2_cpy_chunk_num - 1]);
            
            chunk_gather_copy_kernel(src_keys3, dst_keys, src_index, src_cpy_size, dst_index, size_thread_map, src3_size_per_group, dst_size_per_group, src1_cpy_num+src2_cpy_vector_num, number, bid);
            chunk_gather_copy_kernel(src_values3, dst_values, src_index, src_cpy_size, dst_index, size_thread_map, src3_size_per_group, dst_size_per_group, src1_cpy_num+src2_cpy_vector_num, number, bid);
        }
    }
}



// vector-level gather copy from src and chunk-level scatter to dst
template <typename T>
__device__ __forceinline__ void gather_copy_scatter_kernel(
    T *src,
    T *dst,
    int *src_index,         // src vector-level index, start vector of each chunk
    int *src_cpy_size,      // number of vectors need to be copy for each chunk
    int *dst_index,         // dst chunk-level index
    int *size_thread_map,   // map from copy vector number to maximum thread number
    int src_size_per_group, // data number per group for src
    int dst_size_per_group, // data number per group for dst
    int src_start,          // src start index (vector-level)
    int cpy_chunk_num,      // chunk num need to be copy
    int bid)                // group id
{
    // one CUDA Block (BLOCK_SIZE_CP threads) handles one chunk data each time,
    // each thread copies sizeof(T) Bytes data once,
    // loop over src_index to copy multiple chunks data.
    // if some of the chunk data is no need to copy, threads for those positions just skip them.

    int64_t bid_64 = bid; // need to cast to int64_t to make sure index not overflow.
    int64_t src_base = (bid_64 * src_size_per_group) * DATA_BYTES / sizeof(T);
    src_base += src_start * VECTOR_SIZE_CP;
    int64_t dst_base = (bid_64 * dst_size_per_group) * DATA_BYTES / sizeof(T); 

    int offset_idx = -1;

    // copy chunk by chunk
#pragma unroll
    while (offset_idx < cpy_chunk_num - 1)
    {
        offset_idx++;
        
        int cpy_size = src_cpy_size[offset_idx];        // number of vectors need to be copy for this chunk
        if (threadIdx.x >= size_thread_map[cpy_size])   // skip this thread for this chunk
            continue;
        
        // compute src and dst offset
        int64_t src_offset = src_base + src_index[offset_idx] * VECTOR_SIZE_CP + threadIdx.x;
        int64_t dst_offset = dst_base + dst_index[offset_idx] * BLOCK_SIZE_CP + threadIdx.x;
        // copy data
        dst[dst_offset] = src[src_offset];
    }
}

// gather copy from src and scatter to dst (for both keys & values)
template <typename T> 
__global__ void gather_copy_scatter(
    T *src_keys, T *dst_keys,       // src and dst for keys
    T *src_values, T *dst_values,   // src and dst for values

    int src_size_per_group,         // data number per group for src
    int dst_size_per_group,         // data number per group for dst

    int index_size,                 // index buffer size per group for the following index buffers
    int src_cpy_start,              // start vector index for src

    int *src_index,                 // src vector-level index, shape (group_num, index_size)
    int *src_cpy_size,              // number of vectors need to be copy for each chunk in src, shape (group_num, index_size)
    int *dst_index,                 // dst chunk-level index for src, shape (group_num, index_size)
    int *src_cpy_num                // number of chunks need to be copy for each group in src, shape (group_num)
) {
    extern __shared__ int s[];
    
    int* size_thread_map = s;           // map from copy vector number to maximum thread number, 0~CHUNK_SIZE
    if (threadIdx.x <= CHUNK_SIZE) {    // we assume CHUNK_SIZE <= BLOCK_SIZE_CP
        size_thread_map[threadIdx.x] = threadIdx.x * VECTOR_SIZE_CP;    // copy i vectors need i * VECTOR_SIZE_CP threads
    }
    int* src_offsets = s + CHUNK_SIZE + 1;      // shared memory for src vector-level index
    int* cpy_size = src_offsets + index_size;   // shared memory for src copy vector number for each chunk
    int* dst_offsets = cpy_size + index_size;   // shared memory for dst chunk-level index

    int bid = blockIdx.x;
    int copy_chunk_num = src_cpy_num[bid];      // number of chunks need to be copy

    // transfer indices to shared memory
    int key_offset = bid * index_size;
#pragma unroll
    for (int idx = threadIdx.x; idx < copy_chunk_num; idx += BLOCK_SIZE_CP) {
        src_offsets[idx] = src_index[key_offset + idx];
        cpy_size[idx] = src_cpy_size[key_offset + idx];
        dst_offsets[idx] = dst_index[key_offset + idx];
    }
    __syncthreads();

    gather_copy_scatter_kernel(src_keys, dst_keys, src_offsets, cpy_size, dst_offsets, size_thread_map, src_size_per_group, dst_size_per_group, src_cpy_start, copy_chunk_num, bid);
    gather_copy_scatter_kernel(src_values, dst_values, src_offsets, cpy_size, dst_offsets, size_thread_map, src_size_per_group, dst_size_per_group, src_cpy_start, copy_chunk_num, bid);
}


#endif // COPY_CUH