import re
import six
import csv




filename = "../baselines/spmm_cublas_fp16.txt"
pattern = r"runtime (\d+(\.\d+)?) ms"

reader = open(filename, 'r')
runtime = []

while True:
    line = reader.readline()
    if len(line) == 0:
        break
    line = line.rstrip()
    m = re.search(pattern, line)
    if m:
        runtime.append(m.group(1))

#writer = open('0_spmm_cublas_fp16.txt', 'w')
#for r in runtime:
#    writer.write(str(r) + '\n')
#writer.close()



filename = "../baselines/spmm_cublas_int8.txt"
pattern = r"runtime (\d+(\.\d+)?) ms"

reader = open(filename, 'r')
runtime1 = []

while True:
    line = reader.readline()
    if len(line) == 0:
        break
    line = line.rstrip()
    m = re.search(pattern, line)
    if m:
        runtime1.append(m.group(1))

#writer = open('0_spmm_cublas_int8.txt', 'w')
#for r in runtime1:
#    writer.write(str(r) + '\n')
#writer.close()



filename = "../baselines/spmm_vectorSparse.txt"
pattern = r"runtime (\d+(\.\d+)?) ms"

reader = open(filename, 'r')
runtime2 = []

while True:
    line = reader.readline()
    if len(line) == 0:
        break
    line = line.rstrip()
    m = re.search(pattern, line)
    if m:
        runtime2.append(m.group(1))

#writer = open('0_spmm_vectorSparse.txt', 'w')
#for r in runtime2:
#    writer.write(str(r) + '\n')
#writer.close()



filename = "../baselines/spmm_cusparse_fp16.txt"
pattern = r"runtime (\d+(\.\d+)?) ms"

reader = open(filename, 'r')
runtime3 = []

while True:
    line = reader.readline()
    if len(line) == 0:
        break
    line = line.rstrip()
    m = re.search(pattern, line)
    if m:
        runtime3.append(m.group(1))

#writer = open('0_spmm_cusparse_fp16.txt', 'w')
#for r in runtime3:
#    writer.write(str(r) + '\n')
#writer.close()



filename = "../baselines/spmm_cusparse_int8.txt"
pattern = r"runtime (\d+(\.\d+)?) ms"

reader = open(filename, 'r')
runtime4 = []

while True:
    line = reader.readline()
    if len(line) == 0:
        break
    line = line.rstrip()
    m = re.search(pattern, line)
    if m:
        runtime4.append(m.group(1))

#writer = open('0_spmm_cusparse_int8.txt', 'w')
#for r in runtime4:
#    writer.write(str(r) + '\n')
#writer.close()



filename = "../SpMM/SpMM/spmm_magicube_16b8b.txt"
pattern = r"runtime (\d+(\.\d+)?) ms"

reader = open(filename, 'r')
runtime5 = []

while True:
    line = reader.readline()
    if len(line) == 0:
        break
    line = line.rstrip()
    m = re.search(pattern, line)
    if m:
        runtime5.append(m.group(1))

#writer = open('0_spmm_magicube_16b8b.txt', 'w')
#for r in runtime5:
#    writer.write(str(r) + '\n')
#writer.close()



filename = "../SpMM/SpMM/spmm_magicube_8b8b.txt"
pattern = r"runtime (\d+(\.\d+)?) ms"

reader = open(filename, 'r')
runtime6 = []

while True:
    line = reader.readline()
    if len(line) == 0:
        break
    line = line.rstrip()
    m = re.search(pattern, line)
    if m:
        runtime6.append(m.group(1))

#writer = open('0_spmm_magicube_8b8b.txt', 'w')
#for r in runtime6:
#    writer.write(str(r) + '\n')
#writer.close()



filename = "../SpMM/SpMM/spmm_magicube_8b4b.txt"
pattern = r"runtime (\d+(\.\d+)?) ms"

reader = open(filename, 'r')
runtime7 = []

while True:
    line = reader.readline()
    if len(line) == 0:
        break
    line = line.rstrip()
    m = re.search(pattern, line)
    if m:
        runtime7.append(m.group(1))

#writer = open('0_spmm_magicube_8b4b.txt', 'w')
#for r in runtime7:
#    writer.write(str(r) + '\n')
#writer.close()



filename = "../SpMM/SpMM/spmm_magicube_4b4b.txt"
pattern = r"runtime (\d+(\.\d+)?) ms"

reader = open(filename, 'r')
runtime8 = []

while True:
    line = reader.readline()
    if len(line) == 0:
        break
    line = line.rstrip()
    m = re.search(pattern, line)
    if m:
        runtime8.append(m.group(1))

#writer = open('0_spmm_magicube_4b4b.txt', 'w')
#for r in runtime8:
#    writer.write(str(r) + '\n')
#writer.close()

#header = ['dimN', 'vecLen', 'Sparsity', 'cuBLAS-f16', 'cuBLAS-int8', 'vectorSparse-f16', 'cuSPARSE-f16', 'cuSPARSE-int8', 'Magicube-L16R8', 'Magicube-L8R8', 'Magicube-L8R4', 'Magicube-L4R4']
#
#m = 9216
#k = 1536 
#data_list = [[] * m for i in range(m)]
#
#for i in range(m):
#    if i < m//2:
#        data_list[i].append('N=128')
#    else:
#        data_list[i].append('N=256')
#
#for i in range(m):
#    if i < m//6:
#        data_list[i].append('V=2')
#    elif i < m//6*2:
#        data_list[i].append('V=4')
#    elif i < m//6*3:
#        data_list[i].append('V=8')
#    elif i < m//6*4:
#        data_list[i].append('V=2')
#    elif i < m//6*5:
#        data_list[i].append('V=4')
#    else:
#        data_list[i].append('V=8')
#
#for i in range(6):
#    ii = i*k
#    for j in range(k):
#        if j < k//6:
#            data_list[ii+j].append(0.5)
#        elif j < k//6*2:
#            data_list[ii+j].append(0.7)
#        elif j < k//6*3:
#            data_list[ii+j].append(0.8)
#        elif j < k//6*4:
#            data_list[ii+j].append(0.9)
#        elif j < k//6*5:
#            data_list[ii+j].append(0.95)
#        else:
#            data_list[ii+j].append(0.98)

#m = 9216
#for i in range(m):
#    data_list[i].append(round(float(runtime[i])/float(runtime[i]), 4))
#    data_list[i].append(round(float(runtime[i])/float(runtime1[i]), 4))
#    data_list[i].append(round(float(runtime[i])/float(runtime2[i]), 4))
#    data_list[i].append(round(float(runtime[i])/float(runtime3[i]), 4))
#    data_list[i].append(round(float(runtime[i])/float(runtime4[i]), 4))
#    data_list[i].append(round(float(runtime[i])/float(runtime5[i]), 4))
#    data_list[i].append(round(float(runtime[i])/float(runtime6[i]), 4))
#    data_list[i].append(round(float(runtime[i])/float(runtime7[i]), 4))
#    data_list[i].append(round(float(runtime[i])/float(runtime8[i]), 4))

#with open('0_spmm_all_matrices.csv', 'w', encoding='UTF8') as f:
#    writer = csv.writer(f)
#
#    writer.writerow(header)
#    writer.writerows(data_list)


header = ['algs', 'N', 'V', 'sparsity', 'speedup']
m = 9216
k = 1536 
data_list = [[] * 5 for i in range(m*9)]

for ll in range(9):
    if ll == 0:
        for i in range(m):
            data_list[i + ll*m].append('cuBLAS-f16')
    if ll == 1:
        for i in range(m):
            data_list[i + ll*m].append('cuBLAS-int8')
    if ll == 2:
        for i in range(m):
            data_list[i + ll*m].append('vectorSparse-f16')
    if ll == 3:
        for i in range(m):
            data_list[i + ll*m].append('cuSPARSE-f16')
    if ll == 4:
        for i in range(m):
            data_list[i + ll*m].append('cuSPARSE-int8')
    if ll == 5:
        for i in range(m):
            data_list[i + ll*m].append('Magicube-L16R8')
    if ll == 6:
        for i in range(m):
            data_list[i + ll*m].append('Magicube-L8R8')
    if ll == 7:
        for i in range(m):
            data_list[i + ll*m].append('Magicube-L8R4')
    if ll == 8:
        for i in range(m):
            data_list[i + ll*m].append('Magicube-L4R4')

    for i in range(m):
        if i < m//2:
            data_list[i + ll*m].append('128')
        else:
            data_list[i + ll*m].append('256')
    
    for i in range(m):
        if i < m//6:
            data_list[i + ll*m].append('2')
        elif i < m//6*2:
            data_list[i + ll*m].append('4')
        elif i < m//6*3:
            data_list[i + ll*m].append('8')
        elif i < m//6*4:
            data_list[i + ll*m].append('2')
        elif i < m//6*5:
            data_list[i + ll*m].append('4')
        else:
            data_list[i + ll*m].append('8')
    
    for i in range(6):
        ii = i*k
        for j in range(k):
            if j < k//6:
                data_list[ii+j + ll*m].append(0.5)
            elif j < k//6*2:
                data_list[ii+j + ll*m].append(0.7)
            elif j < k//6*3:
                data_list[ii+j + ll*m].append(0.8)
            elif j < k//6*4:
                data_list[ii+j + ll*m].append(0.9)
            elif j < k//6*5:
                data_list[ii+j + ll*m].append(0.95)
            else:
                data_list[ii+j + ll*m].append(0.98)

    if ll == 0:
        for i in range(m):
            data_list[i + ll*m].append(round(float(runtime[i])/float(runtime[i]), 4))
    if ll == 1:
        for i in range(m):
            data_list[i + ll*m].append(round(float(runtime[i])/float(runtime1[i]), 4))
    if ll == 2:
        for i in range(m):
            data_list[i + ll*m].append(round(float(runtime[i])/float(runtime2[i]), 4))
    if ll == 3:
        for i in range(m):
            data_list[i + ll*m].append(round(float(runtime[i])/float(runtime3[i]), 4))
    if ll == 4:
        for i in range(m):
            data_list[i + ll*m].append(round(float(runtime[i])/float(runtime4[i]), 4))
    if ll == 5:
        for i in range(m):
            data_list[i + ll*m].append(round(float(runtime[i])/float(runtime5[i]), 4))
    if ll == 6:
        for i in range(m):
            data_list[i + ll*m].append(round(float(runtime[i])/float(runtime6[i]), 4))
    if ll == 7:
        for i in range(m):
            data_list[i + ll*m].append(round(float(runtime[i])/float(runtime7[i]), 4))
    if ll == 8:
        for i in range(m):
            data_list[i + ll*m].append(round(float(runtime[i])/float(runtime8[i]), 4))

with open('spmm_all_matrices.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)

    writer.writerow(header)
    writer.writerows(data_list)
