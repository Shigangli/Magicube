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




header = ['S0.9,Seq_l=4096,num_h=8', 'algs', 'Latency(ms)']
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
            data_list[i + ll*m].append(float(runtime[i+16]))
    if ll == 1:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime1[i+16]))
    if ll == 2:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime2[i + 64*1]))
    if ll == 3:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime2[i+8 + 64*1]))
    if ll == 4:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime2[i+16 + 64*1]))
    if ll == 5:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime2[i+24 + 64*1]))
    if ll == 6:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime[i+24]))
    if ll == 7:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime1[i+24]))
    if ll == 8:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime2[i+32 + 64*1]))
    if ll == 9:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime2[i+40 + 64*1]))
    if ll == 10:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime2[i+48 + 64*1]))
    if ll == 11:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime2[i+56 + 64*1]))

with open('n2n_b.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)

    writer.writerow(header)
    writer.writerows(data_list)



header = ['S0.9,Seq_l=8192,num_h=4', 'algs', 'Latency(ms)']
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
            data_list[i + ll*m].append(float(runtime[i+32]))
    if ll == 1:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime1[i+32]))
    if ll == 2:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime2[i + 64*2]))
    if ll == 3:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime2[i+8 + 64*2]))
    if ll == 4:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime2[i+16 + 64*2]))
    if ll == 5:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime2[i+24 + 64*2]))
    if ll == 6:
        for i in range(m):
            data_list[i + ll*m].append(None)
    if ll == 7:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime1[i+40]))
    if ll == 8:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime2[i+32 + 64*2]))
    if ll == 9:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime2[i+40 + 64*2]))
    if ll == 10:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime2[i+48 + 64*2]))
    if ll == 11:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime2[i+56 + 64*2]))

with open('n2n_c.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)

    writer.writerow(header)
    writer.writerows(data_list)



header = ['S0.9,Seq_l=8192,num_h=8', 'algs', 'Latency(ms)']
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
            data_list[i + ll*m].append(float(runtime[i+40]))
    if ll == 1:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime1[i+48]))
    if ll == 2:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime2[i + 64*3]))
    if ll == 3:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime2[i+8 + 64*3]))
    if ll == 4:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime2[i+16 + 64*3]))
    if ll == 5:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime2[i+24 + 64*3]))
    if ll == 6:
        for i in range(m):
            data_list[i + ll*m].append(None)
    if ll == 7:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime1[i+56]))
    if ll == 8:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime2[i+32 + 64*3]))
    if ll == 9:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime2[i+40 + 64*3]))
    if ll == 10:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime2[i+48 + 64*3]))
    if ll == 11:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime2[i+56 + 64*3]))

with open('n2n_d.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)

    writer.writerow(header)
    writer.writerows(data_list)





header = ['S0.95,Seq_l=4096,num_h=4', 'algs', 'Latency(ms)']
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
            data_list[i + ll*m].append(float(runtime1[i + 64]))
    if ll == 2:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime2[i + 64*4]))
    if ll == 3:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime2[i+8 + 64*4]))
    if ll == 4:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime2[i+16 + 64*4]))
    if ll == 5:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime2[i+24 + 64*4]))
    if ll == 6:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime[i+8]))
    if ll == 7:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime1[i+8 + 64]))
    if ll == 8:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime2[i+32 + 64*4]))
    if ll == 9:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime2[i+40 + 64*4]))
    if ll == 10:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime2[i+48 + 64*4]))
    if ll == 11:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime2[i+56 + 64*4]))

with open('n2n_e.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)

    writer.writerow(header)
    writer.writerows(data_list)




header = ['S0.95,Seq_l=4096,num_h=8', 'algs', 'Latency(ms)']
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
            data_list[i + ll*m].append(float(runtime[i+16]))
    if ll == 1:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime1[i+16 + 64]))
    if ll == 2:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime2[i + 64*5]))
    if ll == 3:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime2[i+8 + 64*5]))
    if ll == 4:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime2[i+16 + 64*5]))
    if ll == 5:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime2[i+24 + 64*5]))
    if ll == 6:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime[i+24]))
    if ll == 7:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime1[i+24 + 64]))
    if ll == 8:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime2[i+32 + 64*5]))
    if ll == 9:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime2[i+40 + 64*5]))
    if ll == 10:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime2[i+48 + 64*5]))
    if ll == 11:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime2[i+56 + 64*5]))

with open('n2n_f.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)

    writer.writerow(header)
    writer.writerows(data_list)



header = ['S0.95,Seq_l=8192,num_h=4', 'algs', 'Latency(ms)']
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
            data_list[i + ll*m].append(float(runtime[i+32]))
    if ll == 1:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime1[i+32 + 64]))
    if ll == 2:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime2[i + 64*6]))
    if ll == 3:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime2[i+8 + 64*6]))
    if ll == 4:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime2[i+16 + 64*6]))
    if ll == 5:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime2[i+24 + 64*6]))
    if ll == 6:
        for i in range(m):
            data_list[i + ll*m].append(None)
    if ll == 7:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime1[i+40 + 64]))
    if ll == 8:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime2[i+32 + 64*6]))
    if ll == 9:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime2[i+40 + 64*6]))
    if ll == 10:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime2[i+48 + 64*6]))
    if ll == 11:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime2[i+56 + 64*6]))

with open('n2n_g.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)

    writer.writerow(header)
    writer.writerows(data_list)



header = ['S0.95,Seq_l=8192,num_h=8', 'algs', 'Latency(ms)']
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
            data_list[i + ll*m].append(float(runtime[i+40]))
    if ll == 1:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime1[i+48 + 64]))
    if ll == 2:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime2[i + 64*7]))
    if ll == 3:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime2[i+8 + 64*7]))
    if ll == 4:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime2[i+16 + 64*7]))
    if ll == 5:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime2[i+24 + 64*7]))
    if ll == 6:
        for i in range(m):
            data_list[i + ll*m].append(None)
    if ll == 7:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime1[i+56 + 64]))
    if ll == 8:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime2[i+32 + 64*7]))
    if ll == 9:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime2[i+40 + 64*7]))
    if ll == 10:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime2[i+48 + 64*7]))
    if ll == 11:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime2[i+56 + 64*7]))

with open('n2n_h.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)

    writer.writerow(header)
    writer.writerows(data_list)
