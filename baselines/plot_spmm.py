import numpy as np
import matplotlib.pyplot as plt
import argparse
import csv
import os
from file_name_server import get_file_name, extract_duration_ncu, geometric_mean
import matplotlib.ticker as ticker

# Args
parser = argparse.ArgumentParser(description='plot the acceleration')

parser.add_argument('--start', type=int, default=0, help='the starting benchmark to run')
parser.add_argument('--end', type=int, default=1130, help='the ending benchmark to run')
parser.add_argument('--dimK', type=int, default=-1, help='the dimension k of the benchmark')
parser.add_argument('--sort', action='store_true', help='sort the csr list')
parser.add_argument('--bm', choices=['rn50', 'transformer'], default='rn50', help='the benchmark to plot')
parser.add_argument('--combo', action='store_true', help='plot all Ks in the same plot')

args = parser.parse_args()

bm_list = open('/users/shigang/gitrepo/dlmc/%s_matrices.txt' % args.bm, 'r')
lines = bm_list.readlines()

def extract_duration_set(v, k, kernel, list_):
    if kernel == 'wmma':
        suffix = '_wmma'
    elif kernel == 'cuda':
        suffix = '_cuda'
    else:
        suffix = '_dense'
    
    suffix_dense = '_dense_sort'

    if args.sort:
        suffix += '_sort'
        
    for i in np.arange(args.start, args.end):
        benchmark = lines[i][:-6]
        file_kernel = get_file_name('/users/shigang/gitrepo/dlmc/%s' % benchmark, k, v, kernel, True, 'spmm', 'None', 'half', False)
        file_dense = get_file_name('/users/shigang/gitrepo/dlmc/%s' % benchmark, k, v, 'dense', True, 'spmm', 'None', 'half', False)
        # file_kernel = './csv/dlmc/%s_k%d_v%d.csv' % (benchmark + suffix, args.dimK, v)
        # file_dense = './csv/dlmc/%s_k%d_v%d.csv' % (benchmark + suffix_dense, args.dimK, v)
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

fig, axs = plt.subplots(2, 2 * len(Ks), figsize=(16,5.5))

for idx, k in enumerate(Ks):
    # dense_v1 = [[], [], [], [], [], []]
    cuda_v1 = [[], [], [], [], [], []]
    extract_duration_set(1, k, 'sputnik', cuda_v1)

    # dense_v2 = [[], [], [], [], [], []]
    cuda_v2 = [[], [], [], [], [], []]
    wmma_v2 = [[], [], [], [], [], []]
    bell_v2 = [[], [], [], [], [], []]

    extract_duration_set(2, k, 'cuda', cuda_v2)
    extract_duration_set(2, k, 'wmma', wmma_v2)
    extract_duration_set(2, k, 'bell', bell_v2)

    # dense_v4 = [[], [], [], [], [], []]
    cuda_v4 = [[], [], [], [], [], []]
    wmma_v4 = [[], [], [], [], [], []]
    bell_v4 = [[], [], [], [], [], []]

    extract_duration_set(4, k, 'cuda', cuda_v4)
    extract_duration_set(4, k, 'wmma', wmma_v4)
    extract_duration_set(4, k, 'bell', bell_v4)

    # dense_v8 = [[], [], [], [], [], []]
    cuda_v8 = [[], [], [], [], [], []]
    wmma_v8 = [[], [], [], [], [], []]
    bell_v8 = [[], [], [], [], [], []]

    extract_duration_set(8, k, 'cuda', cuda_v8)
    extract_duration_set(8, k, 'wmma', wmma_v8)
    extract_duration_set(8, k, 'bell', bell_v8)

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
            widths=0.25)

    axs[0, 0 + 2 * idx].plot([0.5, 6.5],[1, 1], color='purple')
    c1_p = plot(axs[0, 0 + 2 * idx], 'steelblue', 0, cuda_v1, 'cuda')
    

    axs[0, 0 + 2 * idx].set_xticks([1, 2, 3, 4, 5, 6])
    axs[0, 0 + 2 * idx].set_xticklabels([0.5, 0.7, 0.8, 0.9, 0.95, 0.98])
    axs[0, 0 + 2 * idx].legend([c1_p["boxes"][0]], ['fpu'], loc='upper left')
    # axs[0, 0 + 2 * idx].set_ylabel('Speedup over cublasHgemm')
    axs[0, 0 + 2 * idx].set_title('V=1, N=%d' % k)
    axs[0, 0 + 2 * idx].yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

    axs[0, 1 + 2 * idx].plot([0.5, 7],[1, 1], color='purple')
    c2_p = plot(axs[0, 1 + 2 * idx], 'steelblue', 0, cuda_v2, 'cuda')
    b2_p = plot(axs[0, 1 + 2 * idx], 'forestgreen', 0.3, bell_v2, 'blocked_ell')
    w2_p = plot(axs[0, 1 + 2 * idx], 'lightcoral', 0.6, wmma_v2, 'wmma')

    axs[0, 1 + 2 * idx].legend([c2_p["boxes"][0], b2_p["boxes"][0], w2_p["boxes"][0]], ['fpu', 'blocked-ELL', 'mma'], loc='upper left')
    axs[0, 1 + 2 * idx].set_xticks([1, 2, 3, 4, 5, 6])
    axs[0, 1 + 2 * idx].set_xticklabels([0.5, 0.7, 0.8, 0.9, 0.95, 0.98])
    axs[0, 1 + 2 * idx].set_title('V=2, N=%d' % k)

    axs[1, 0 + 2 * idx].plot([0.5, 7],[1, 1], color='purple')
    c4_p = plot(axs[1, 0 + 2 * idx], 'steelblue', 0, cuda_v4, 'cuda')
    b4_p = plot(axs[1, 0 + 2 * idx], 'forestgreen', 0.3, bell_v4, 'blocked_ell')
    w4_p = plot(axs[1, 0 + 2 * idx], 'lightcoral', 0.6, wmma_v4, 'wmma')

    axs[1, 0 + 2 * idx].legend([c4_p["boxes"][0], b4_p["boxes"][0], w4_p["boxes"][0]], ['fpu', 'blocked-ELL', 'mma'], loc='upper left')
    axs[1, 0 + 2 * idx].set_xticks([1, 2, 3, 4, 5, 6])
    axs[1, 0 + 2 * idx].set_xticklabels([0.5, 0.7, 0.8, 0.9, 0.95, 0.98])
    # axs[1, 0 + 2 * idx].set_xlabel('Sparsity')
    # axs[1, 0 + 2 * idx].set_ylabel('Speedup over cublasHgemm')
    axs[1, 0 + 2 * idx].set_title('V=4, N=%d' % k)

    axs[1, 1 + 2 * idx].plot([0.5, 7],[1, 1], color='purple')
    c8_p =plot(axs[1, 1 + 2 * idx], 'steelblue', 0, cuda_v8, 'cuda')
    b8_p = plot(axs[1, 1 + 2 * idx], 'forestgreen', 0.3, bell_v8, 'blocked_ell')
    w8_p = plot(axs[1, 1 + 2 * idx], 'lightcoral', 0.6, wmma_v8, 'wmma')

    axs[1, 1 + 2 * idx].legend([c8_p["boxes"][0], b8_p["boxes"][0], w8_p["boxes"][0]], ['fpu', 'blocked-ELL', 'mma'], loc='upper left')
    axs[1, 1 + 2 * idx].set_xticks([1, 2, 3, 4, 5, 6])
    axs[1, 1 + 2 * idx].set_xticklabels([0.5, 0.7, 0.8, 0.9, 0.95, 0.98])
    # axs[1, 1 + 2 * idx].set_xlabel('Sparsity')
    axs[1, 1 + 2 * idx].set_title('V=8, N=%d' % k)

fig.tight_layout()

fig.add_subplot(111, frame_on=False)
plt.tick_params(labelcolor="none", bottom=False, left=False)
plt.ylabel("Speedup over cuBLASHgemm", fontsize=13)
plt.xlabel("Sparsity", fontsize=13)
plt.subplots_adjust(hspace=0.2, wspace=0.2)

if args.combo:
    fig.savefig('./spmm_speedup_%s_combo.pdf' % (args.bm), bbox_inches='tight')
else:
    fig.savefig('./spmm_speedup_%s_k%d.pdf' % (args.bm, args.dimK), bbox_inches='tight')

with open('./spmm_geo.csv', 'w') as outfile:
    writer = csv.writer(outfile)
    for r in geo_rows:
        writer.writerow(r)
