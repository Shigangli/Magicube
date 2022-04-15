import numpy as np
import matplotlib.pyplot as plt
import argparse
import csv
import os

# Args
parser = argparse.ArgumentParser(description='plot the acceleration')

parser.add_argument('--start', type=int, default=0, help='the starting benchmark to run')
parser.add_argument('--end', type=int, default=1130, help='the ending benchmark to run')
parser.add_argument('--dimK', type=int, default=-1, help='the dimension k of the benchmark')
parser.add_argument('--sort', action='store_true', help='sort the csr list')

args = parser.parse_args()

bm_list = open('/users/shigang/gitrepo/dlmc/rn50_matrices.txt', 'r')
lines = bm_list.readlines()

# function that collects the result
def extract_duration_ncu(file):
    if os.path.exists(file):
        with open(file, 'r') as csvfile:
            csvreader = csv.reader(csvfile)

            unit = 'unknown'
            dur_accumulate = 0
            for row in csvreader:
                if len(row) >= 3 and "Duration" in row[-4]:
                    unit = row[-3]
                    try:
                        dur = float(row[-2])
                        if unit == 'second':
                            dur *= 1000
                        elif unit == 'usecond':
                            dur /= 1000
                        elif unit == 'nsecond':
                            dur /= 1e+6
                        else:
                            print('unknown unit')
                        dur_accumulate += dur
                    except:
                        print(file)
                        return -1.0
            return dur_accumulate
    else:
        print('file %s does not exist' % file)

def extract_duration_set(v, kernel, list_, precision, sddmm=False):
    suffix = '_%s' % kernel

    suffix_dense = '_dense'

    suffix += '_%s' % precision
    suffix_dense += '_%s' % precision
    
    if args.sort and not sddmm:
        suffix += '_sort'
        suffix_dense += '_sort'

    if sddmm:
        job = 'sddmm'
    else:
        job = 'spmm'

    for i in np.arange(args.start, args.end):
        benchmark = lines[i][:-6]
        file_kernel = './csv/%s/dlmc/%s_k%d_v%d.csv' % (job, benchmark + suffix, args.dimK, v)
        file_dense = './csv/%s/dlmc/%s_k%d_v%d.csv' % (job, benchmark + suffix_dense, args.dimK, v)
        # if kernel == 'dense':
        #     print(file_kernel)
        #     print(file_dense)
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
                print("undefined sparsity")

# SpMM
ell_v2 = [[], [], [], [], [], []]
ell_v4 = [[], [], [], [], [], []]
ell_v8 = [[], [], [], [], [], []]
ell_v16 = [[], [], [], [], [], []]

extract_duration_set(2, 'blocked_ell', ell_v2, 'half')
extract_duration_set(4, 'blocked_ell', ell_v4, 'half')
extract_duration_set(8, 'blocked_ell', ell_v8, 'half')
extract_duration_set(16, 'blocked_ell', ell_v16, 'half')


def plot(ax, color, bias, data, label='nn'):
    return ax.boxplot(data, positions=[1 + bias, 2 + bias, 3 + bias, 4 + bias, 5 + bias, 6 + bias], notch=True, patch_artist=True,
        boxprops=dict(facecolor=color),
        capprops=dict(color=color),
        whiskerprops=dict(color=color),
        flierprops=dict(color=color, markeredgecolor=color),
        medianprops=dict(color='black'),
        widths=0.35)

"""
fig, axs = plt.subplots(2, 2, figsize=(6.5, 4.5))

axs[0, 0].plot([0.5, 6.5],[1, 1], color='purple')
ell2_p = plot(axs[0, 0], 'steelblue', 0, ell_v2, 'sputnik')
# axs[0, 0].legend([sf_p["boxes"][0], cf_p["boxes"][0]], [''], loc='upper left')
axs[0, 0].set_xticks([1, 2, 3, 4, 5, 6])
axs[0, 0].set_xticklabels([0.5, 0.7, 0.8, 0.9, 0.95, 0.98])
axs[0, 0].set_title('Block-Size=2')
# axs[0, 0].set_ylabel('Speedup over cublasHgemm')


axs[0, 1].plot([0.5, 6.5],[1, 1], color='purple')
ell4_p = plot(axs[0, 1], 'steelblue', 0, ell_v4, 'sputnik')
# axs[0, 1].legend([sh_p["boxes"][0], ch_p["boxes"][0]], ['sputnik', 'cusparse'], loc='upper left')
axs[0, 1].set_xticks([1, 2, 3, 4, 5, 6])
axs[0, 1].set_xticklabels([0.5, 0.7, 0.8, 0.9, 0.95, 0.98])
axs[0, 1].set_title('Block-Size=4')
# axs[0, 1].set_ylabel('Speedup over cublasHgemm')


axs[1, 0].plot([0.5, 6.5],[1, 1], color='purple')
ell8_p = plot(axs[1, 0], 'steelblue', 0, ell_v8, 'sputnik')
# df_d = plot(axs[1, 0], 'purple', 0.5, dense_single, 'dense')
# axs[1, 0].legend([sf_d["boxes"][0], cf_d["boxes"][0]], ['sputnik', 'cusparse'], loc='upper left')
axs[1, 0].set_xticks([1, 2, 3, 4, 5, 6])
axs[1, 0].set_xticklabels([0.5, 0.7, 0.8, 0.9, 0.95, 0.98])
axs[1, 0].set_title('Block-Size=8')
# axs[1, 0].set_xlabel('Sparsity')
# axs[1, 0].set_ylabel('Speedup over cublasHgemm')


axs[1, 1].plot([0.5, 6.5],[1, 1], color='purple')
ell16_p = plot(axs[1, 1], 'steelblue', 0, ell_v16, 'sputnik')
# dh_d = plot(axs[1, 1], 'purple', 0.5, dense_half, 'dense')
# axs[1, 1].legend([sh_d["boxes"][0]], ['sputnik'], loc='upper left')
axs[1, 1].set_xticks([1, 2, 3, 4, 5, 6])
axs[1, 1].set_xticklabels([0.5, 0.7, 0.8, 0.9, 0.95, 0.98])
axs[1, 1].set_title('Block-Size=16')
# axs[1, 1].set_xlabel('Sparsity')
# axs[1, 1].set_ylabel('Speedup over cublasHgemm')

plt.subplots_adjust(hspace=0.32)

fig.add_subplot(111, frame_on=False)
plt.tick_params(labelcolor="none", bottom=False, left=False)
plt.ylabel("Speedup over cuBLASHgemm")
plt.xlabel("Sparsity")

fig.savefig('./blocked_ell_speedup.pdf', bbox_inches='tight')
"""

fig, axs = plt.subplots(1, 3, figsize=(7, 2.5))

"""
axs[0].plot([0.5, 6.5],[1, 1], color='purple')
ell2_p = plot(axs[0], 'steelblue', 0, ell_v2, 'sputnik')
# axs[0, 0].legend([sf_p["boxes"][0], cf_p["boxes"][0]], [''], loc='upper left')
axs[0].set_xticks([1, 2, 3, 4, 5, 6])
axs[0].set_xticklabels([0.5, 0.7, 0.8, 0.9, 0.95, 0.98])
axs[0].set_title('Block-Size=2')
# axs[0, 0].set_ylabel('Speedup over cublasHgemm')
"""

axs[0].plot([0.5, 6.5],[1, 1], color='purple')
ell4_p = plot(axs[0], 'steelblue', 0, ell_v4, 'sputnik')
# axs[0, 1].legend([sh_p["boxes"][0], ch_p["boxes"][0]], ['sputnik', 'cusparse'], loc='upper left')
axs[0].set_xticks([1, 2, 3, 4, 5, 6])
axs[0].set_xticklabels([0.5, 0.7, 0.8, 0.9, 0.95, 0.98])
axs[0].set_title('Block-Size=4')
# axs[0, 1].set_ylabel('Speedup over cublasHgemm')


axs[1].plot([0.5, 6.5],[1, 1], color='purple')
ell8_p = plot(axs[1], 'steelblue', 0, ell_v8, 'sputnik')
# df_d = plot(axs[1, 0], 'purple', 0.5, dense_single, 'dense')
# axs[1, 0].legend([sf_d["boxes"][0], cf_d["boxes"][0]], ['sputnik', 'cusparse'], loc='upper left')
axs[1].set_xticks([1, 2, 3, 4, 5, 6])
axs[1].set_xticklabels([0.5, 0.7, 0.8, 0.9, 0.95, 0.98])
axs[1].set_title('Block-Size=8')
# axs[1, 0].set_xlabel('Sparsity')
# axs[1, 0].set_ylabel('Speedup over cublasHgemm')


axs[2].plot([0.5, 6.5],[1, 1], color='purple')
ell16_p = plot(axs[2], 'steelblue', 0, ell_v16, 'sputnik')
# dh_d = plot(axs[1, 1], 'purple', 0.5, dense_half, 'dense')
# axs[1, 1].legend([sh_d["boxes"][0]], ['sputnik'], loc='upper left')
axs[2].set_xticks([1, 2, 3, 4, 5, 6])
axs[2].set_xticklabels([0.5, 0.7, 0.8, 0.9, 0.95, 0.98])
axs[2].set_title('Block-Size=16')
# axs[1, 1].set_xlabel('Sparsity')
# axs[1, 1].set_ylabel('Speedup over cublasHgemm')

plt.subplots_adjust(wspace=0.15)

fig.add_subplot(111, frame_on=False)
plt.tick_params(labelcolor="none", bottom=False, left=False)
plt.ylabel("Speedup over cuBLASHgemm", fontsize=11)
plt.xlabel("Sparsity", fontsize=11)

fig.savefig('./blocked_ell_speedup.pdf', bbox_inches='tight')
