import numpy as np
import matplotlib.pyplot as plt
import csv
from file_name_server import get_file_name
import os

# function that collects the memory usage
def extract_mem_ncu(file):
    if os.path.exists(file):
        with open(file, 'r') as csvfile:
            csvreader = csv.reader(csvfile)

            for row in csvreader:
                if len(row) >= 2 and "byte" in row[-3]:
                    return int(row[-1])
        print('No result found')
            
    print('file %s does not exist' % file)


bell_v2 = []
bell_v4 = []
bell_v8 = []

bells = [bell_v2, bell_v4, bell_v8]

mma_v2 = []
mma_v4 = []
mma_v8 = []

mmas = [mma_v2, mma_v4, mma_v8]

for idx, v in enumerate([2, 4, 8]):
    for sparse in ['0.5', '0.7', '0.8', '0.9', '0.95', '0.98']:
        bm = '/users/shigang/gitrepo/dlmc/rn50/magnitude_pruning/%s/bottleneck_projection_block_group_projection_block_group4.smtx' % sparse
        output_file = get_file_name(bm, 256, v, 'bell', False, 'spmm', 'blank', 'half', True)
        bells[idx].append(extract_mem_ncu(output_file))
        output_file = get_file_name(bm, 256, v, 'wmma', True, 'spmm', 'blank', 'half', True)
        mmas[idx].append(extract_mem_ncu(output_file))

fig, ax = plt.subplots(1, 3, figsize=(8, 2))

labels = ['0.5', '0.7', '0.8', '0.9', '0.95', '0.98']
x = np.arange(len(labels))
width = 0.35
ax[0].bar(x - width/2, bell_v2, width, label='Blocked Ell', color='steelblue')
ax[0].bar(x + width/2, mma_v2, width, label='Vector-Sparse', color='lightcoral')
ax[0].set_ylabel('Bytes L2\$->L1\$')
ax[0].set_xticks(x)
ax[0].set_xticklabels(labels)
ax[0].set_xlabel('Sparsity')
ax[0].legend(loc = 'upper right')

ax[1].bar(x - width/2, bell_v4, width, label='Blocked Ell', color='steelblue')
ax[1].bar(x + width/2, mma_v4, width, label='Vector-Sparse', color='lightcoral')
ax[1].set_xticks(x)
ax[1].set_xticklabels(labels)
ax[1].set_xlabel('Sparsity')
ax[1].legend(loc = 'upper right')

ax[2].bar(x - width/2, bell_v8, width, label='Blocked Ell', color='steelblue')
ax[2].bar(x + width/2, mma_v8, width, label='Vector-Sparse', color='lightcoral')
ax[2].set_xticks(x)
ax[2].set_xticklabels(labels)
ax[2].set_xlabel('Sparsity')
ax[2].legend(loc = 'upper right')

fig.savefig('./mem_l2_l1.pdf', bbox_inches='tight')
