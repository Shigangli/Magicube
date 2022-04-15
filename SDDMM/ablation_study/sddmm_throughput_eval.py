import os
import argparse

# Args
parser = argparse.ArgumentParser(description='lauch the sddmm benchmarks')

args = parser.parse_args()

dataset_dir = os.environ.get('dataset_dir')
sparsities = ['0.5', '0.7', '0.8', '0.9', '0.95', '0.98']
vec_lens = [2, 4, 8]

print("****************** SDDMM basic *******************")
for sparsity in sparsities:
    for vec_len in vec_lens:
        matrix = '%s/rn50/random_pruning/%s/bottleneck_2_block_group3_5_1.smtx' % (dataset_dir, sparsity)
        cmd = './SDDMM_basic/sddmm_benchmark %s 512 %d 1 1 1 16 16' % (matrix, vec_len)
        os.system(cmd)

for sparsity in sparsities:
    for vec_len in vec_lens:
        matrix = '%s/rn50/random_pruning/%s/bottleneck_2_block_group3_5_1.smtx' % (dataset_dir, sparsity)
        cmd = './SDDMM_basic/sddmm_benchmark %s 512 %d 1 1 1 8 8' % (matrix, vec_len)
        os.system(cmd)

for sparsity in sparsities:
    for vec_len in vec_lens:
        matrix = '%s/rn50/random_pruning/%s/bottleneck_2_block_group3_5_1.smtx' % (dataset_dir, sparsity)
        cmd = './SDDMM_basic/sddmm_benchmark %s 512 %d 1 1 1 4 4' % (matrix, vec_len)
        os.system(cmd)


print("****************** SDDMM with LHS prefetch *******************")
for sparsity in sparsities:
    for vec_len in vec_lens:
        matrix = '%s/rn50/random_pruning/%s/bottleneck_2_block_group3_5_1.smtx' % (dataset_dir, sparsity)
        cmd = './SDDMM_lhs_pref/sddmm_benchmark %s 512 %d 1 1 1 16 16' % (matrix, vec_len)
        os.system(cmd)

for sparsity in sparsities:
    for vec_len in vec_lens:
        matrix = '%s/rn50/random_pruning/%s/bottleneck_2_block_group3_5_1.smtx' % (dataset_dir, sparsity)
        cmd = './SDDMM_lhs_pref/sddmm_benchmark %s 512 %d 1 1 1 8 8' % (matrix, vec_len)
        os.system(cmd)

for sparsity in sparsities:
    for vec_len in vec_lens:
        matrix = '%s/rn50/random_pruning/%s/bottleneck_2_block_group3_5_1.smtx' % (dataset_dir, sparsity)
        cmd = './SDDMM_lhs_pref/sddmm_benchmark %s 512 %d 1 1 1 4 4' % (matrix, vec_len)
        os.system(cmd)
