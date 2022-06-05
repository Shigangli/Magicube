import re
import six
import csv


filename = "../baselines/sddmm_cublas_fp16.txt"
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

#writer = open('0_sddmm_cublas_fp16.txt', 'w')
#for r in runtime:
#    writer.write(str(r) + '\n')
#writer.close()


filename = "../baselines/sddmm_cublas_int8.txt"
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

#writer = open('0_sddmm_cublas_int8.txt', 'w')
#for r in runtime1:
#    writer.write(str(r) + '\n')
#writer.close()


filename = "../baselines/sddmm_vectorSparse.txt"
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

#writer = open('0_sddmm_vectorSparse.txt', 'w')
#for r in runtime2:
#    writer.write(str(r) + '\n')
#writer.close()


filename = "../SDDMM/SDDMM/sddmm_magicube_16b16b.txt"
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

#writer = open('0_sddmm_magicube_16b16b.txt', 'w')
#for r in runtime3:
#    writer.write(str(r) + '\n')
#writer.close()


filename = "../SDDMM/SDDMM/sddmm_magicube_8b8b.txt"
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

#writer = open('0_sddmm_magicube_8b8b.txt', 'w')
#for r in runtime4:
#    writer.write(str(r) + '\n')
#writer.close()


filename = "../SDDMM/SDDMM/sddmm_magicube_4b4b.txt"
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

#writer = open('0_sddmm_magicube_4b4b.txt', 'w')
#for r in runtime5:
#    writer.write(str(r) + '\n')
#writer.close()

header = ['algs', 'K', 'V', 'sparsity', 'speedup']
m = 9216
k = 1536 
data_list = [[] * 5 for i in range(m*6)]

for ll in range(6):
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
            data_list[i + ll*m].append('Magicube-L16R16')
    if ll == 4:
        for i in range(m):
            data_list[i + ll*m].append('Magicube-L8R8')
    if ll == 5:
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

with open('sddmm_all_matrices.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)

    writer.writerow(header)
    writer.writerows(data_list)
