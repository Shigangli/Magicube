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

