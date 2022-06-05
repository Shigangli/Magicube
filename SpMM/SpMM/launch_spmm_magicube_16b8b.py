import os
import argparse
import numpy as np

# Args
parser = argparse.ArgumentParser(description='lauch the spmm benchmarks')

#parser.add_argument('--dimN', type=int, default=256, help="the dimension N of the benchmark")
#parser.add_argument('--dimV', type=int, default=8, help="vector length")
#parser.add_argument('--sparsity', choices=['50', '70', '80', '90', '95', '98'], default='70', help='sparsity of the matrix')
#parser.add_argument('--preA', type=int, default=8, help="number of bits for A")
#parser.add_argument('--preB', type=int, default=8, help="number of bits for B")
args = parser.parse_args()

dataset_dir = os.environ.get('dataset_dir')
sparsities = ['50', '70', '80', '90', '95', '98']
dimNs = [128, 256]
vec_lens = [2, 4, 8]

for dimN in dimNs:
    for vec_len in vec_lens:
        for sparsity in sparsities:
            print("dimN: ", dimN, "vec_len: ", vec_len, "sparsity: ", sparsity)
        
            matrix_list = open('./eval_matrices/s%s.txt' % sparsity, 'r')
            lines = matrix_list.readlines()
            for i in range(len(lines)):
            #for i in range(1):
                matrix = '%s/%s' % (dataset_dir, lines[i][:-1])
                cmd = './spmm_benchmark %s %d %d 0 1 0 1 16 8' % (matrix, dimN, vec_len)
                os.system(cmd)

