import os
import argparse

# Args
parser = argparse.ArgumentParser(description='lauch the sddmm benchmarks')

#parser.add_argument('--dimK', type=int, default=256, help="the dimension N of the benchmark")
#parser.add_argument('--dimV', type=int, default=8, help="vector length")
#parser.add_argument('--sparsity', choices=['50', '70', '80', '90', '95', '98'], default='70', help='sparsity of the matrix')
#parser.add_argument('--preA', type=int, default=8, help="number of bits for A")
#parser.add_argument('--preB', type=int, default=8, help="number of bits for B")
args = parser.parse_args()

dataset_dir = '/users/shigang/gitrepo/dlmc'
sparsities = ['0.5', '0.7', '0.8', '0.9', '0.95', '0.98']
vec_lens = [2, 4, 8]

for sparsity in sparsities:
    for vec_len in vec_lens:
        matrix = '%s/rn50/random_pruning/%s/bottleneck_2_block_group3_5_1.smtx' % (dataset_dir, sparsity)
        cmd = './sddmm_benchmark %s 512 %d 1 1 1 16 16' % (matrix, vec_len)
        os.system(cmd)

for sparsity in sparsities:
    for vec_len in vec_lens:
        matrix = '%s/rn50/random_pruning/%s/bottleneck_2_block_group3_5_1.smtx' % (dataset_dir, sparsity)
        cmd = './sddmm_benchmark %s 512 %d 1 1 1 8 8' % (matrix, vec_len)
        os.system(cmd)

for sparsity in sparsities:
    for vec_len in vec_lens:
        matrix = '%s/rn50/random_pruning/%s/bottleneck_2_block_group3_5_1.smtx' % (dataset_dir, sparsity)
        cmd = './sddmm_benchmark %s 512 %d 1 1 1 4 4' % (matrix, vec_len)
        os.system(cmd)
