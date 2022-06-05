import re
import six
import csv


filename = "../SpMM/ablation_study/spmm_abl_study.txt"
pattern = r"TOP/s: (\d+(\.\d+)?)"

reader = open(filename, 'r')
tops = []

while True:
    line = reader.readline()
    if len(line) == 0:
        break
    line = line.rstrip()
    m = re.search(pattern, line)
    if m:
        tops.append(m.group(1))

#writer = open('0_spmm_abl_study.txt', 'w')
#for r in tops:
#    writer.write(str(r) + '\n')
#writer.close()


header = ['opts', 'configs', 'TOP/s']
m = 16 
data_list = [[] * 3 for i in range(m*4)]

for ll in range(4):
    if ll == 0:
        for i in range(m):
            data_list[i + ll*m].append('basic')
    if ll == 1:
        for i in range(m):
            data_list[i + ll*m].append('conflict-free')
    if ll == 2:
        for i in range(m):
            data_list[i + ll*m].append('conflict-free+prefetch')
    if ll == 3:
        for i in range(m):
            data_list[i + ll*m].append('conflict-free+prefetch+col_idx_shuffle')

    data_list[0 + ll*m].append('V=2,L16-R8,Sp=0.7')
    data_list[1 + ll*m].append('V=8,L16-R8,Sp=0.7')
    data_list[2 + ll*m].append('V=2,L16-R8,Sp=0.9')
    data_list[3 + ll*m].append('V=8,L16-R8,Sp=0.9')
    data_list[4 + ll*m].append('V=2,L8-R8,Sp=0.7')
    data_list[5 + ll*m].append('V=8,L8-R8,Sp=0.7')
    data_list[6 + ll*m].append('V=2,L8-R8,Sp=0.9')
    data_list[7 + ll*m].append('V=8,L8-R8,Sp=0.9')
    data_list[8 + ll*m].append('V=2,L8-R4,Sp=0.7')
    data_list[9 + ll*m].append('V=8,L8-R4,Sp=0.7')
    data_list[10 + ll*m].append('V=2,L8-R4,Sp=0.9')
    data_list[11 + ll*m].append('V=8,L8-R4,Sp=0.9')
    data_list[12 + ll*m].append('V=2,L4-R4,Sp=0.7')
    data_list[13 + ll*m].append('V=8,L4-R4,Sp=0.7')
    data_list[14 + ll*m].append('V=2,L4-R4,Sp=0.9')
    data_list[15 + ll*m].append('V=8,L4-R4,Sp=0.9')

    if ll == 3:
        for i in range(8):
            data_list[i + ll*m].append(None)
        for i in range(8):
            data_list[i + ll*m + 8].append(float(tops[i + ll*m]))
    else:
        for i in range(m):
            data_list[i + ll*m].append(float(tops[i + ll*m]))


with open('spmm_abl_study.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)

    writer.writerow(header)
    writer.writerows(data_list)
