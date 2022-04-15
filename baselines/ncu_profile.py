import os
import argparse
import string
import csv


# Args
parser = argparse.ArgumentParser(description='profile the SpMM Kernel')

parser.add_argument('--ncu', default=os.environ.get('NCU_PATH'),
                            help='Path to nsight compute')
parser.add_argument('--bm', default='/users/shigang/gitrepo/dlmc/rn50/random_pruning/0.8/bottleneck_2_block_group3_5_1.smtx',
                            help='Path to benchmark')
parser.add_argument('--dimK', '-k', type=int, default=256, help='the dimension K of the benchmark')
parser.add_argument('--dimV', '-v', type=int, default=1, help='the vector length')
parser.add_argument('--kernel', choices=['wmma', 'cuda', 'dense', 'sputnik', 'cusparse', 'bell'], help='select a kernel to profile')
parser.add_argument('--sort', action='store_true', help='sort the csr list')
parser.add_argument('--prof', action='store_true', help='profile the kernel')
parser.add_argument('--func', action='store_true', help='do functional verification')
parser.add_argument('--job', choices=['spmm', 'sddmm'], help='choose the job to launch')
parser.add_argument('--sddmm_alg', choices=['wmma', 'mma_reg', 'mma_shfl', 'mma_arch'], default='mma_reg',
                    help='the algorithm used for wmmaSddmm')
parser.add_argument('--precision', choices=['half', 'single'], help='the precesion of the arithmatics')
parser.add_argument('--print', action='store_true', help='print the execution time')
parser.add_argument('--mem', action='store_true', help='do memory profile')

args = parser.parse_args()

#
# Set up the input args and the suffix of the output file
#

# Select the kernel to profile
if args.kernel == 'wmma':
    ker = 0
    suffix = '_wmma'
    sparse = 1
elif args.kernel == 'cuda':
    ker = 1
    suffix = '_cuda'
    sparse = 1
elif args.kernel == 'sputnik':
    ker = 2
    suffix = '_sputnik'
    sparse = 1
elif args.kernel == 'cusparse':
    ker = 3
    suffix = '_cusparse'
    sparse = 1
elif args.kernel == 'bell':
    ker = 0
    suffix = '_blocked_ell'
    sparse = 2
else:
    ker = 0
    suffix = '_dense'
    sparse = 0

# Select the precision
if args.precision == 'half':
    mixed = 1
    suffix += '_half'
else:
    mixed = 0
    suffix += '_single'

if args.sort:
    suffix += '_sort'
    sort = 1
else:
    sort = 0

# If func = 1, the output of the kernel will be compared with the result on host
if args.func:
    func = 1
else:
    func = 0

# If the job is SpMM
if args.job == 'spmm':
    exe_cmd = './spmm_benchmark %s %d %d %d %d %d %d %d' % (args.bm, args.dimK, args.dimV, ker, sort, func, sparse, mixed)

    if args.prof:
        if args.mem:
            output_file = './csv/mem/spmm/%s_k%d_v%d.csv' % (args.bm.replace('/users/shigang/gitrepo/', '').replace('.smtx', '') + suffix, args.dimK, args.dimV)
            exe_cmd = '--metrics l1tex__m_xbar2l1tex_read_bytes_mem_lg_op_ld.sum %s' % exe_cmd
        else:
            output_file = './csv/spmm/%s_k%d_v%d.csv' % (args.bm.replace('/users/shigang/gitrepo/', '').replace('.smtx', '') + suffix, args.dimK, args.dimV)
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        cmd = '%s -f --csv --cache-control all --profile-from-start 0 --log-file %s %s' % (args.ncu, output_file, exe_cmd)
    else:
        cmd = exe_cmd

    print(cmd)
    os.system(cmd)
# Else if the job is SDDMM
else:
    if args.sddmm_alg == 'wmma':
        alg = 0
    elif args.sddmm_alg == 'mma_reg':
        alg = 1
    elif args.sddmm_alg == 'mma_shfl':
        alg = 2
    elif args.sddmm_alg == 'mma_arch':
        alg = 3
    else:
        alg = 1
    
    if args.kernel == 'wmma':
        suffix += '_%s' % args.sddmm_alg
    exe_cmd = './sddmm_benchmark %s %d %d %d %d %d %d %d %d' % (args.bm, args.dimK, args.dimV, ker, alg, sort, func, sparse, mixed)

    if args.prof:
        output_file = './csv/sddmm/%s_k%d_v%d.csv' % (args.bm.replace('/users/shigang/gitrepo/', '').replace('.smtx', '') + suffix, args.dimK, args.dimV)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        cmd = '%s -f --csv --cache-control all --profile-from-start 0 --log-file %s %s' % (args.ncu, output_file, exe_cmd)
    else:
        cmd = exe_cmd
    
    print(cmd)
    os.system(cmd)



# Get duration
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

if args.print:
    dur = extract_duration_ncu(output_file)
    print("Duration: %.4f ms" % dur)

