import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import csv

from numpy.core.fromnumeric import squeeze

def log_name(feature, seq_len, sparsity):
    return './csv/f_%d_seq_%d_sp_%.2f.csv' % (feature, seq_len, sparsity)

def launch_profile(feature, seq_len, sparsity):
    prof_cmd = 'nvprof --csv --log-file %s --profile-from-start off -f ' % log_name(feature, seq_len, sparsity)
    py_cmd = 'python3 ./sparse_encoder.py --embed_dim %d --seq_len %d --sparsity %.2f' % (feature * 8, seq_len, sparsity)

    cmd = prof_cmd + py_cmd
    print(cmd)
    os.system(cmd)

unit_scale = {
    's': float(1e+3),
    'ms': float(1), 
    'us': float(1e-3),
    'ns': float(1e-6)
}

def process_profile(feature, seq_len, sparsity):
    log = log_name(feature, seq_len, sparsity)
    timing = {
        'QK^T': 0.,
        'Softmax': 0.,
        'AV': 0.,
        'All': 0.,
        'else': 0.,
        'sp QK^T': 0.,
        'sp Softmax': 0.,
        'sp AV': 0.,
        'sp All': 0.,
        'sp else': 0.
    }
    with open(log, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        nvtx_domain = None
        unit = None

        for row in csvreader:
            if len(row) > 0:
                # Processing the AV domain of the kernel
                if 'Range "AV"' in row[0]:
                    nvtx_domain = 'AV'
                    # print(nvtx_domain)
                elif 'Range "MultiheadAttention"' in row[0]:
                    nvtx_domain = 'All'
                    # print(nvtx_domain)
                elif 'Range "QK^T"' in row[0]:
                    nvtx_domain = 'QK^T'
                    # print(nvtx_domain)
                elif 'Range "Softmax' in row[0]:
                    nvtx_domain = "Softmax"
                    # print(nvtx_domain)
                elif 'Range "sp AV"' in row[0]:
                    nvtx_domain = 'sp AV'
                    # print(nvtx_domain)
                elif 'Range "spMultiheadAttention"' in row[0]:
                    nvtx_domain = 'sp All'
                    # print(nvtx_domain)
                elif 'Range "sp QK^T"' in row[0]:
                    nvtx_domain = 'sp QK^T'
                    # print(nvtx_domain)
                elif 'Range "sp Softmax' in row[0]:
                    nvtx_domain = "sp Softmax"
                    # print(nvtx_domain)
                elif 'API calls' in row[0] and nvtx_domain is not None:
                    nvtx_domain = None
                    # print('Switch off')
            if len(row) > 2 and nvtx_domain is not None:
                if row[1] == '%':
                    unit = row[-4]
                if "GPU activities" in row[0]:
                    time = float(row[3]) * float(row[4]) / 5. * unit_scale[unit]
                    timing[nvtx_domain] += time
        timing['else'] = timing['All'] - timing['AV'] - timing['Softmax'] - timing['QK^T']
        timing['sp else'] = timing['sp All'] - timing['sp AV'] - timing['sp Softmax'] - timing['sp QK^T']

        print("(%d, %d, %.2f): speedup: %.4f" % (feature, seq_len, sparsity, timing['All'] / timing['sp All']))
        return timing
                

def single_plot(feature, seq_len, axes, x, y):
    timings = []
    # Step 1: run the three experiments
    for sp in [0.9, 0.95, 0.98]:
        # launch_profile(feature, seq_len, sp)
        timings.append(process_profile(feature, seq_len, sp))
    
    timing_qk = [timings[0]['QK^T']]
    timing_softmax = [timings[0]['Softmax']]
    timing_av = [timings[0]['AV']]
    timing_else = [timings[0]['else']]

    for timing in timings:
        timing_qk.append(timing['sp QK^T'])
        timing_softmax.append(timing['sp Softmax'])
        timing_av.append(timing['sp AV'])
        timing_else.append(timing['sp else'])

    sp_labels = ['dense', '0.9', '0.95', '0.98']
    x_ = np.arange(len(sp_labels))
    width = 0.4

    axes[x, y].bar(x_, timing_qk, width, label=r'$QK^T\odot C$', color='steelblue')
    bottom = timing_qk
    axes[x, y].bar(x_, timing_softmax, width, bottom=bottom, label='Softmax', color='lightcoral')
    for idx in range(len(bottom)):
        bottom[idx] += timing_softmax[idx]
    axes[x, y].bar(x_, timing_av, width, bottom=bottom, label=r'$AV$', color='forestgreen')
    for idx in range(len(bottom)):
        bottom[idx] += timing_av[idx]
    axes[x, y].bar(x_, timing_else, width, bottom=bottom, label='Others', color='grey')
    axes[x, y].set_xticks(x_)
    axes[x, y].set_xticklabels(sp_labels)
    axes[x, y].set_title('l=%d, k=%d' % (seq_len, feature))

def single_plot_1d(feature, seq_len, axes, x):
    timings = []
    # Step 1: run the three experiments
    for sp in [0.9, 0.95, 0.98]:
        # launch_profile(feature, seq_len, sp)
        timings.append(process_profile(feature, seq_len, sp))
    
    timing_qk = [timings[0]['QK^T']]
    timing_softmax = [timings[0]['Softmax']]
    timing_av = [timings[0]['AV']]
    timing_else = [timings[0]['else']]

    for timing in timings:
        timing_qk.append(timing['sp QK^T'])
        timing_softmax.append(timing['sp Softmax'])
        timing_av.append(timing['sp AV'])
        timing_else.append(timing['sp else'])

    sp_labels = ['dense', '0.9', '0.95', '0.98']
    x_ = np.arange(len(sp_labels))
    width = 0.4

    axes[x].bar(x_, timing_qk, width, label=r'$QK^T\odot C$', color='steelblue')
    bottom = timing_qk
    axes[x].bar(x_, timing_softmax, width, bottom=bottom, label='Softmax', color='lightcoral')
    for idx in range(len(bottom)):
        bottom[idx] += timing_softmax[idx]
    axes[x].bar(x_, timing_av, width, bottom=bottom, label=r'$AV$', color='forestgreen')
    for idx in range(len(bottom)):
        bottom[idx] += timing_av[idx]
    axes[x].bar(x_, timing_else, width, bottom=bottom, label='Others', color='grey')
    axes[x].set_xticks(x_)
    axes[x].set_xticklabels(sp_labels)
    axes[x].set_title('l=%d, k=%d' % (seq_len, feature))


"""
fig, axes = plt.subplots(3, 3, figsize = (8, 8))

seq_lens = [2048, 4096, 8192]
features = [64, 128, 256]

for idx_s, seq_len in enumerate(seq_lens):
    for idx_f, feature in enumerate(features):
        single_plot(feature, seq_len, axes, idx_s, idx_f)

fig.add_subplot(111, frame_on=False)
plt.tick_params(labelcolor="none", bottom=False, left=False)
plt.ylabel("Latency (ms)", fontsize=13)
plt.xlabel("Sparsity", fontsize=13)

axes[0, 1].legend(ncol=4, bbox_to_anchor = (0.7,0.35,1,1))

plt.subplots_adjust(hspace=0.3)


fig.savefig('./atten_speedup.pdf', bbox_inches='tight')
"""

fig, axes = plt.subplots(1, 4, figsize = (8, 2))
single_plot_1d(64, 2048, axes, 0)
single_plot_1d(64, 4096, axes, 1)
single_plot_1d(64, 8192, axes, 2)
single_plot_1d(256, 8192, axes, 3)
fig.add_subplot(111, frame_on=False)
plt.tick_params(labelcolor="none", bottom=False, left=False)
plt.ylabel("Latency (ms)", fontsize=13)
plt.xlabel("Sparsity", fontsize=13)

axes[1].legend(ncol=4, bbox_to_anchor = (1.8,0.4,1,1))

plt.subplots_adjust(hspace=0.3)


fig.savefig('./atten_speedup_v2.pdf', bbox_inches='tight')
