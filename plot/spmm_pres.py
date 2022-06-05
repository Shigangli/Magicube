import re
import six


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

writer = open('0_spmm_pres.txt', 'w')
for r in tops:
    writer.write(str(r) + '\n')
writer.close()
