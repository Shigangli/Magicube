import torch
from scipy.sparse import random
import numpy as np
from torch._C import dtype


def static_random_mask(m, n, sparsity):
    # Step 1: generate the sparse mask
    csr_mask = random(m=m, n=n, density = 1. - sparsity, format='csr')
    column_indices = csr_mask.indices
    row_offsets = csr_mask.indptr

    # Step 2: sort row by length
    row_length = []

    for i in range(m):
        row_length.append(row_offsets[i+1] - row_offsets[i])

    row_indices = np.argsort(row_length)

    # Step 3: push all the vectors to GPU
    column_indices = torch.tensor(column_indices, dtype=torch.int32, device='cuda')
    row_offsets = torch.tensor(row_offsets, dtype=torch.int32, device='cuda')
    row_indices = torch.tensor(row_indices, dtype=torch.int32, device='cuda')

    return column_indices, row_offsets, row_indices


def static_random_mask_aligned(m, n, sparsity, mma_k_dim):

    csr_mask = random(m=m, n=n, density = 1. - sparsity, format='csr')
    column_indices = csr_mask.indices
    row_offsets = csr_mask.indptr

    aligned_num_item = 0
    aligned_row_offsets = np.zeros(m*2, dtype='int32')
    aligned_row_offsets[0] = aligned_num_item
    for i in range(1, m+1):
        num_item = row_offsets[i] - row_offsets[i-1]
        aligned_num_item += int((num_item + mma_k_dim - 1) / mma_k_dim) * mma_k_dim
        if i != m:
            aligned_row_offsets[i*2] = aligned_num_item
        aligned_row_offsets[i*2-1] = aligned_row_offsets[i*2-2] + num_item

    row_length = []

    for i in range(m):
        row_length.append(row_offsets[i+1] - row_offsets[i])

    row_indices = np.argsort(row_length)

    aligned_col_indices = np.zeros(aligned_num_item, dtype='int32')
    aligned_col_indices_shuffle = np.zeros(aligned_num_item, dtype='int32')

    for i in range(aligned_num_item):
        aligned_col_indices[i] = -1
        aligned_col_indices_shuffle[i] = -1

    for i in range(1, m+1):
        offset_begin = row_offsets[i-1]
        offset_end = row_offsets[i]
        for j in range(offset_begin, offset_end):
            aligned_col_indices[aligned_row_offsets[(i-1)*2] + j - offset_begin] = column_indices[j]

    chunks = int(aligned_num_item/8)

    for i in range(chunks):
        for j in range(8):
            aligned_col_indices_shuffle[i*8 + (j%2)*4 + int(j/2)] = aligned_col_indices[i*8 + j]

    column_indices = torch.tensor(aligned_col_indices, dtype=torch.int32, device='cuda')
    column_indices_shuffle = torch.tensor(aligned_col_indices_shuffle, dtype=torch.int32, device='cuda')
    row_offsets = torch.tensor(aligned_row_offsets, dtype=torch.int32, device='cuda')
    row_indices = torch.tensor(row_indices, dtype=torch.int32, device='cuda')

    return column_indices, column_indices_shuffle, row_offsets, row_indices, aligned_num_item


def csr2dense(column_indices, row_offsets, values, m, n, vec_length, val=0.):
    column_indices_cpu = column_indices.cpu().detach().numpy()
    row_offsets_cpu = row_offsets.cpu().detach().numpy()
    values_cpu = values.cpu().detach().numpy()

    value_dense = np.full(shape=(m * vec_length, n), fill_value = val, dtype=np.half)

    for m_vec in range(m):
        row_nnz = row_offsets_cpu[m_vec + 1] - row_offsets_cpu[m_vec]
        for j in range(row_nnz):
            for v in range(vec_length):
                row_idx = m_vec * vec_length + v
                col_idx = column_indices_cpu[row_offsets_cpu[m_vec] + j]
                value_dense[row_idx][col_idx] = values_cpu[(row_offsets_cpu[m_vec] + j) * vec_length + v]
    
    value_dense = torch.tensor(value_dense, dtype=torch.float16, device='cuda')
    return value_dense


def batched_csr2dense(column_indices, row_offsets, values, m, n, vec_length, batch_size, val=0.):
    column_indices_cpu = column_indices.cpu().detach().numpy()
    row_offsets_cpu = row_offsets.cpu().detach().numpy()
    values_cpu = values.cpu().detach().numpy()
    value_dense = np.full(shape=(batch_size, m * vec_length, n), fill_value = val, dtype=np.half)

    for m_vec in range(m):
        row_nnz = row_offsets_cpu[m_vec + 1] - row_offsets_cpu[m_vec]
        for j in range(row_nnz):
            for v in range(vec_length):
                row_idx = m_vec * vec_length + v
                col_idx = column_indices_cpu[row_offsets_cpu[m_vec] + j]
                for b in range(batch_size):
                    value_dense[b][row_idx][col_idx] = values_cpu[b][(row_offsets_cpu[m_vec] + j) * vec_length + v]
    
    value_dense = torch.tensor(value_dense, dtype=torch.float16, device='cuda')
    return value_dense

