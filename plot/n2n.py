import re
import six
import csv


filename = "../end2end_eval/sparse_transformer_baselines/pytorch_n2n.txt"
pattern = r"runtime (\d+(\.\d+)?) milliseconds"

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

#writer = open('0_pytorch_n2n.txt', 'w')
#for r in runtime:
#    writer.write(str(r) + '\n')
#writer.close()


filename = "../end2end_eval/sparse_transformer_baselines/vectorSparse_n2n.txt"
pattern = r"runtime (\d+(\.\d+)?) milliseconds"

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

#writer = open('0_vectorSparse_n2n.txt', 'w')
#for r in runtime1:
#    writer.write(str(r) + '\n')
#writer.close()


filename = "../end2end_eval/sparse_transformer_magicube/magicube_n2n.txt"
pattern = r"runtime (\d+(\.\d+)?) milliseconds"

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

#3writer = open('0_magicube_n2n.txt', 'w')
#3for r in runtime2:
#3    writer.write(str(r) + '\n')
#3writer.close()


header = ['S0.9,Seq_l=4096,num_h=4', 'algs', 'Latency(ms)']
m = 8 
data_list = [[] * 3 for i in range(m*12)]

for ll in range(12):
    if ll < 6:
        for i in range(m):
            data_list[i + ll*m].append('batchsize=2')
    else:
        for i in range(m):
            data_list[i + ll*m].append('batchsize=8')


    if ll == 0:
        for i in range(m):
            data_list[i + ll*m].append('Pytorch-fp16')
    if ll == 1:
        for i in range(m):
            data_list[i + ll*m].append('vectorSparse-fp16')
    if ll == 2:
        for i in range(m):
            data_list[i + ll*m].append('Magicube-4b4b')
    if ll == 3:
        for i in range(m):
            data_list[i + ll*m].append('Magicube-8b4b')
    if ll == 4:
        for i in range(m):
            data_list[i + ll*m].append('Magicube-8b8b')
    if ll == 5:
        for i in range(m):
            data_list[i + ll*m].append('Magicube-16b8b')
    if ll == 6:
        for i in range(m):
            data_list[i + ll*m].append('Pytorch-fp16')
    if ll == 7:
        for i in range(m):
            data_list[i + ll*m].append('vectorSparse-fp16')
    if ll == 8:
        for i in range(m):
            data_list[i + ll*m].append('Magicube-4b4b')
    if ll == 9:
        for i in range(m):
            data_list[i + ll*m].append('Magicube-8b4b')
    if ll == 10:
        for i in range(m):
            data_list[i + ll*m].append('Magicube-8b8b')
    if ll == 11:
        for i in range(m):
            data_list[i + ll*m].append('Magicube-16b8b')

    if ll == 0:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime[i]))
    if ll == 1:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime1[i]))
    if ll == 2:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime2[i]))
    if ll == 3:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime2[i+8]))
    if ll == 4:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime2[i+16]))
    if ll == 5:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime2[i+24]))
    if ll == 6:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime[i+8]))
    if ll == 7:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime1[i+8]))
    if ll == 8:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime2[i+32]))
    if ll == 9:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime2[i+40]))
    if ll == 10:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime2[i+48]))
    if ll == 11:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime2[i+56]))



with open('n2n_a.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)

    writer.writerow(header)
    writer.writerows(data_list)
