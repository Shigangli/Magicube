import os
import argparse
import numpy as np

# Args
parser = argparse.ArgumentParser(description='lauch the sddmm benchmarks')

parser.add_argument('--start', type=int, default=0, help="the starting benchmark to run")
parser.add_argument('--end', type=int, default=1130, help="the ending benchmark to run")
parser.add_argument('--dimK', type=int, default=-1, help="the dimension k of the benchmark")
parser.add_argument('--dimV', type=int, default=1, help="the dimension v of the benchmark")
parser.add_argument('--dimK_mul', type=int, default=1, help="scare the dimK by a constant factor")
parser.add_argument('--kernel', choices=['wmma', 'cuda', 'dense', 'sputnik', 'cusparse', 'bell'], help='select a kernel to profile')
parser.add_argument('--sort', action='store_true', help='sort the csr list')
parser.add_argument('--func', action='store_true', help='do function verification')
parser.add_argument('--job', choices=['spmm', 'sddmm'], help='choose the job to launch')
parser.add_argument('--bm', choices=['rn50', 'transformer'], default='rn50', help='the benchmark to run')
parser.add_argument('--sddmm_alg', choices=['wmma', 'mma_reg', 'mma_shfl', 'mma_arch'], default='mma_reg',
                    help='the algorithm used for wmmaSddmm')
parser.add_argument('--precision', choices=['half', 'single'], help='the precesion of the arithmatics')

args = parser.parse_args()

# This collects the dimension K from the benchmark. 
# The benchmark - K is organized as a dictionary to be indexed later.
k_list = open('/users/shigang/gitrepo/dlmc/rn50_batchsizes.txt', 'r')
k_lines = k_list.readlines()
k_dict = {}
for l in k_lines:
    layer, dimk = l.split(',')
    k_dict[layer] = int(dimk)

# collect the benchmark names
bm_list = open('/users/shigang/gitrepo/dlmc/%s_matrices.txt' % args.bm, 'r')
lines = bm_list.readlines()

# A for loop traverses all the benchmarks within the declared range
for i in np.arange(args.start, args.end):
    # get the benchmark name
    benchmark = lines[i][:-6]
    # When an arbitrary dimK is provided, use it
    # Otherwise, use the dimK in the benchmark
    if args.dimK <= 0:
        dimK = k_dict[benchmark.split('/')[-1]]
    else:
        dimK = args.dimK
    
    # the dimK can be scaled by args.dimK_mul
    # It is more useful when the dimK in the benchmark is used
    dimK *= args.dimK_mul

    # complete the benchmark path
    benchmark = '/users/shigang/gitrepo/dlmc/%s.smtx' % benchmark
    
    # switch between spmm and sddmm
    if args.job == 'spmm':
        cmd = 'python3 ncu_profile.py --bm %s -k %d -v %d --kernel %s --prof --job spmm --precision %s' % (benchmark, args.dimK, args.dimV, args.kernel, args.precision)
    else:
        cmd = 'python3 ncu_profile.py --bm %s -k %d -v %d --kernel %s --prof --job sddmm --sddmm_alg %s --precision %s' % (benchmark, args.dimK, args.dimV, args.kernel, args.sddmm_alg, args.precision)
    if args.func: cmd += ' --func'
    if args.sort: cmd += ' --sort'
    # print(cmd)
    os.system(cmd)
    
