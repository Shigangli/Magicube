import os
import argparse
import numpy as np

# Args
parser = argparse.ArgumentParser(description='lauch the spmm benchmarks')

#parser.add_argument('--dimK', type=int, default=256, help="the dimension N of the benchmark")
#parser.add_argument('--dimV', type=int, default=8, help="vector length")
#parser.add_argument('--sparsity', choices=['50', '70', '80', '90', '95', '98'], default='70', help='sparsity of the matrix')
#parser.add_argument('--preA', type=int, default=8, help="number of bits for A")
#parser.add_argument('--preB', type=int, default=8, help="number of bits for B")
args = parser.parse_args()

sparsities = ['50', '70', '80', '90', '95', '98']
dimKs = [128, 256]
vec_lens = [2, 4, 8]

for dimK in dimKs:
    for vec_len in vec_lens:
        for sparsity in sparsities:
            print("dimK: ", dimK, "vec_len: ", vec_len, "sparsity: ", sparsity)
        
            matrix_list = open('/users/shigang/gitrepo/dlmc/s%s.txt' % sparsity, 'r')
            lines = matrix_list.readlines()
            #print("number_of_matrix: ", len(lines))
            #for i in range(1):
            for i in range(len(lines)):
                matrix = '/users/shigang/gitrepo/dlmc/%s' % lines[i][:-1]
                #print(matrix)
                cmd = './sddmm_benchmark %s %d %d 1 0 1 4 4' % (matrix, dimK, vec_len)
                os.system(cmd)

