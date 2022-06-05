import re
import six
import csv


filename = "../SpMM/SpMM/spmm_pres.txt"
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

#writer = open('0_spmm_pres.txt', 'w')
#for r in tops:
#    writer.write(str(r) + '\n')
#writer.close()

header = ['pres', 'configs', 'TOP/s']
m = 18 
data_list = [[] * 3 for i in range(m*7)]

for ll in range(7):
    if ll == 0:
        for i in range(m):
            data_list[i + ll*m].append('L16-R16')
    if ll == 1:
        for i in range(m):
            data_list[i + ll*m].append('L16-R8')
    if ll == 2:
        for i in range(m):
            data_list[i + ll*m].append('L8-R8')
    if ll == 3:
        for i in range(m):
            data_list[i + ll*m].append('L16-R4')
    if ll == 4:
        for i in range(m):
            data_list[i + ll*m].append('L12-R4')
    if ll == 5:
        for i in range(m):
            data_list[i + ll*m].append('L8-R4')
    if ll == 6:
        for i in range(m):
            data_list[i + ll*m].append('L4-R4')

    data_list[0 + ll*m].append('V2,S0.5')
    data_list[1 + ll*m].append('V4,S0.5')
    data_list[2 + ll*m].append('V8,S0.5')
    data_list[3 + ll*m].append('V2,S0.7')
    data_list[4 + ll*m].append('V4,S0.7')
    data_list[5 + ll*m].append('V8,S0.7')
    data_list[6 + ll*m].append('V2,S0.8')
    data_list[7 + ll*m].append('V4,S0.8')
    data_list[8 + ll*m].append('V8,S0.8')
    data_list[9 + ll*m].append('V2,S0.9')
    data_list[10 + ll*m].append('V4,S0.9')
    data_list[11 + ll*m].append('V8,S0.9')
    data_list[12 + ll*m].append('V2,S0.95')
    data_list[13 + ll*m].append('V4,S0.95')
    data_list[14 + ll*m].append('V8,S0.95')
    data_list[15 + ll*m].append('V2,S0.98')
    data_list[16 + ll*m].append('V4,S0.98')
    data_list[17 + ll*m].append('V8,S0.98')

    for i in range(m):
        data_list[i + ll*m].append(float(tops[i + ll*m]))


with open('spmm_pres.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)

    writer.writerow(header)
    writer.writerows(data_list)
