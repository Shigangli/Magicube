import numpy as np
import matplotlib.pyplot as plt
import argparse
import csv
import os
from file_name_server import get_file_name, extract_duration_ncu, geometric_mean

# Args
parser = argparse.ArgumentParser(description='plot the acceleration')

parser.add_argument('--start', type=int, default=0, help='the starting benchmark to run')
parser.add_argument('--end', type=int, default=1130, help='the ending benchmark to run')
parser.add_argument('--dimK', type=int, default=-1, help='the dimension k of the benchmark')
parser.add_argument('--sort', action='store_true', help='sort the csr list')
parser.add_argument('--bm', choices=['rn50', 'transformer'], default='rn50', help='the benchmark to plot')
parser.add_argument('--combo', action='store_true', help='plot all the configurations')

args = parser.parse_args()

bm_list = open('/users/shigang/gitrepo/dlmc/%s_matrices.txt' % args.bm, 'r')
lines = bm_list.readlines()

def extract_duration_set(v, k, kernel, list_, alg):
    for i in np.arange(args.start, args.end):
        benchmark = lines[i][:-6]
        file_kernel = get_file_name('/users/shigang/gitrepo/dlmc/%s' % benchmark, k, v, kernel, args.sort, 'sddmm', alg, 'half', False)
        file_dense = get_file_name('/users/shigang/gitrepo/dlmc/%s' % benchmark, k, v, 'dense', args.sort, 'sddmm', 'None', 'half', False)
        dur_kernel = extract_duration_ncu(file_kernel)
        dur_dense = extract_duration_ncu(file_dense)
        if dur_kernel > 0:
            dur_kernel = dur_dense / dur_kernel
            if ('0.5' in benchmark):
                list_[0].append(dur_kernel)
            elif('0.7' in benchmark):
                list_[1].append(dur_kernel)
            elif('0.8' in benchmark):
                list_[2].append(dur_kernel)
            elif('0.95' in benchmark):
                list_[4].append(dur_kernel)
            elif('0.98' in benchmark):
                list_[5].append(dur_kernel)
            elif('0.9' in benchmark):
                list_[3].append(dur_kernel)
            else:
                # print("undefined sparsity")
                continue

geo_rows = []

if args.combo:
    Ks = [64, 128, 256]
else:
    Ks = [args.dimK]

fig, axs = plt.subplots(4, len(Ks), figsize = (16, 8))

for idx, k in enumerate(Ks):
    # dense_v1 = [[], [], [], [], [], []]
    cuda_v1 = [[], [], [], [], [], []]
    extract_duration_set(1, k, 'cuda', cuda_v1, 'mma_reg')

    # dense_v2 = [[], [], [], [], [], []]
    cuda_v2 = [[], [], [], [], [], []]
    wmma_v2 = [[], [], [], [], [], []]
    mma_reg_v2 = [[], [], [], [], [], []]
    mma_shfl_v2 = [[], [], [], [], [], []]
    mma_arch_v2 = [[], [], [], [], [], []]

    extract_duration_set(2, k, 'cuda', cuda_v2, 'mma_reg')
    extract_duration_set(2, k, 'wmma', wmma_v2, 'wmma')
    extract_duration_set(2, k, 'wmma', mma_reg_v2, 'mma_reg')
    extract_duration_set(2, k, 'wmma', mma_shfl_v2, 'mma_shfl')
    extract_duration_set(2, k, 'wmma', mma_arch_v2, 'mma_arch')

    # dense_v4 = [[], [], [], [], [], []]
    cuda_v4 = [[], [], [], [], [], []]
    wmma_v4 = [[], [], [], [], [], []]
    mma_reg_v4 = [[], [], [], [], [], []]
    mma_shfl_v4 = [[], [], [], [], [], []]
    mma_arch_v4 = [[], [], [], [], [], []]

    extract_duration_set(4, k, 'cuda', cuda_v4, 'mma_reg')
    extract_duration_set(4, k, 'wmma', wmma_v4, 'wmma')
    extract_duration_set(4, k, 'wmma', mma_reg_v4, 'mma_reg')
    extract_duration_set(4, k, 'wmma', mma_shfl_v4, 'mma_shfl')
    extract_duration_set(4, k, 'wmma', mma_arch_v4, 'mma_arch')

    # dense_v8 = [[], [], [], [], [], []]
    cuda_v8 = [[], [], [], [], [], []]
    wmma_v8 = [[], [], [], [], [], []]
    mma_reg_v8 = [[], [], [], [], [], []]
    mma_shfl_v8 = [[], [], [], [], [], []]
    mma_arch_v8 = [[], [], [], [], [], []]

    extract_duration_set(8, k, 'cuda', cuda_v8, 'mma_reg')
    extract_duration_set(8, k, 'wmma', wmma_v8, 'wmma')
    extract_duration_set(8, k, 'wmma', mma_reg_v8, 'mma_reg')
    extract_duration_set(8, k, 'wmma', mma_shfl_v8, 'mma_shfl')
    extract_duration_set(8, k, 'wmma', mma_arch_v8, 'mma_arch')

    def plot(ax, color, bias, data, label='nn'):
        geo_mean = geometric_mean(data)
        geo_rows.append([label] + geo_mean)
        ax.plot([1, 2, 3, 4, 5, 6], geo_mean, color=color)
        return ax.boxplot(data, positions=[1 + bias, 2 + bias, 3 + bias, 4 + bias, 5 + bias, 6 + bias], notch=True, patch_artist=True,
            boxprops=dict(facecolor=color),
            capprops=dict(color=color),
            whiskerprops=dict(color=color),
            flierprops=dict(color=color, markeredgecolor=color),
            medianprops=dict(color='black'),
            widths=0.1)
            
    axs[0, idx].grid(True)
    axs[1, idx].grid(True)
    axs[2, idx].grid(True)
    axs[3, idx].grid(True)

    axs[0, idx].plot([0.5, 6.5],[1, 1], color='purple')
    c1_p = plot(axs[0, idx], 'steelblue', 0, cuda_v1, 'cuda')

    axs[0, idx].set_xticks([1, 2, 3, 4, 5, 6])
    axs[0, idx].set_xticklabels([0.5, 0.7, 0.8, 0.9, 0.95, 0.98])
    # axs[0, 0].legend([c1_p["boxes"][0]], ['cuda'], loc='upper left')
    # axs[0, 0].set_ylabel('Speedup over cublasHgemm')
    axs[0, idx].set_title('V=1, K=%d' % k)

    axs[1, idx].plot([0.5, 7],[1, 1], color='purple')
    c2_p = plot(axs[1, idx], 'steelblue', 0, cuda_v2, 'cuda')
    w2_p = plot(axs[1, idx], 'lightcoral', 0.2, wmma_v2, 'wmma')
    w2_reg_p = plot(axs[1, idx], 'forestgreen', 0.4, mma_reg_v2, 'mma (reg)')
    w2_shfl_p = plot(axs[1, idx], 'mediumslateblue', 0.6, mma_shfl_v2, 'mma (shfl)')
    w2_arch_p = plot(axs[1, idx], 'orange', 0.8, mma_arch_v2, 'mma (arch)')

    # axs[0, 1].legend([c2_p["boxes"][0], w2_p["boxes"][0], w2_reg_p["boxes"][0], w2_shfl_p["boxes"][0], w2_arch_p["boxes"][0]], ['cuda', 'wmma', 'mma (reg)', 'mma (shfl)', 'mma (arch)'], loc='upper left', ncol=2)
    axs[1, idx].set_xticks([1, 2, 3, 4, 5, 6])
    axs[1, idx].set_xticklabels([0.5, 0.7, 0.8, 0.9, 0.95, 0.98])
    axs[1, idx].set_title('V=2, K=%d' % k)

    axs[2, idx].plot([0.5, 7],[1, 1], color='purple')
    c4_p = plot(axs[2, idx], 'steelblue', 0, cuda_v4, 'cuda')
    w4_p = plot(axs[2, idx], 'lightcoral', 0.2, wmma_v4, 'wmma')
    w4_reg_p = plot(axs[2, idx], 'forestgreen', 0.4, mma_reg_v4, 'mma (reg)')
    w4_shfl_p = plot(axs[2, idx], 'mediumslateblue', 0.6, mma_shfl_v4, 'mma (shfl)')
    w4_arch_p = plot(axs[2, idx], 'orange', 0.8, mma_arch_v4, 'mma (arch)')

    # axs[1, 0].legend([c4_p["boxes"][0], w4_p["boxes"][0], w4_reg_p["boxes"][0], w4_shfl_p["boxes"][0], w4_arch_p["boxes"][0]], ['cuda', 'wmma', 'mma (reg)', 'mma (shfl)', 'mma (arch)'], loc='upper left')
    axs[2, idx].set_xticks([1, 2, 3, 4, 5, 6])
    axs[2, idx].set_xticklabels([0.5, 0.7, 0.8, 0.9, 0.95, 0.98])
    axs[2, idx].set_title('V=4, K=%d' % k)
    # axs[1, 0].set_xlabel('Sparsity')
    # axs[1, 0].set_ylabel('Speedup over cublasHgemm')


    axs[3, idx].plot([0.5, 7],[1, 1], color='purple')
    c8_p = plot(axs[3, idx], 'steelblue', 0, cuda_v8, 'cuda')
    w8_p = plot(axs[3, idx], 'lightcoral', 0.2, wmma_v8, 'wmma')
    w8_reg_p = plot(axs[3, idx], 'forestgreen', 0.4, mma_reg_v8, 'mma (reg)')
    w8_shfl_p = plot(axs[3, idx], 'mediumslateblue', 0.6, mma_shfl_v8, 'mma (shfl)')
    w8_arch_p = plot(axs[3, idx], 'orange', 0.8, mma_arch_v8, 'mma (arch)')

    if idx == 0: axs[0, 1].legend([c8_p["boxes"][0], w8_p["boxes"][0], w8_reg_p["boxes"][0], w8_shfl_p["boxes"][0], w8_arch_p["boxes"][0]], ['fpu', 'wmma', 'mma (reg)', 'mma (shfl)', 'mma (arch)'], loc='upper left', ncol=5, bbox_to_anchor = (-0.5,0.5,1,1))
    axs[3, idx].set_xticks([1, 2, 3, 4, 5, 6])
    axs[3, idx].set_xticklabels([0.5, 0.7, 0.8, 0.9, 0.95, 0.98])
    # axs[1, 1].set_xlabel('Sparsity')
    axs[3, idx].set_title('V=8, K=%d' % k)

plt.subplots_adjust(hspace=0.35)

fig.add_subplot(111, frame_on=False)
plt.tick_params(labelcolor="none", bottom=False, left=False)
plt.ylabel("Speedup over cuBLASHgemm", fontsize=13)
plt.xlabel("Sparsity", fontsize=13)

if args.combo:
    fig.savefig('./sddmm_speedup_%s_combo.pdf' % (args.bm), bbox_inches='tight')
else:
    fig.savefig('./sddmm_speedup_%s_k%d.pdf' % (args.bm, args.dimK), bbox_inches='tight')

with open('./sddmm_geo.csv', 'w') as outfile:
    writer = csv.writer(outfile)
    for r in geo_rows:
        writer.writerow(r)
