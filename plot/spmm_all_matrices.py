import re
import six


filename = "../baselines/spmm_cublas_fp16.txt"
pattern = r"runtime (\d+(\.\d+)?) ms"

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

writer = open('0_spmm_cublas_fp16.txt', 'w')
for r in tops:
    writer.write(str(r) + '\n')
writer.close()



filename = "../baselines/spmm_cublas_int8.txt"
pattern = r"runtime (\d+(\.\d+)?) ms"

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

writer = open('0_spmm_cublas_int8.txt', 'w')
for r in tops:
    writer.write(str(r) + '\n')
writer.close()



filename = "../baselines/spmm_vectorSparse.txt"
pattern = r"runtime (\d+(\.\d+)?) ms"

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

writer = open('0_spmm_vectorSparse.txt', 'w')
for r in tops:
    writer.write(str(r) + '\n')
writer.close()



filename = "../baselines/spmm_cusparse_fp16.txt"
pattern = r"runtime (\d+(\.\d+)?) ms"

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

writer = open('0_spmm_cusparse_fp16.txt', 'w')
for r in tops:
    writer.write(str(r) + '\n')
writer.close()



filename = "../baselines/spmm_cusparse_int8.txt"
pattern = r"runtime (\d+(\.\d+)?) ms"

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

writer = open('0_spmm_cusparse_int8.txt', 'w')
for r in tops:
    writer.write(str(r) + '\n')
writer.close()



filename = "../SpMM/SpMM/spmm_magicube_16b8b.txt"
pattern = r"runtime (\d+(\.\d+)?) ms"

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

writer = open('0_spmm_magicube_16b8b.txt', 'w')
for r in tops:
    writer.write(str(r) + '\n')
writer.close()



filename = "../SpMM/SpMM/spmm_magicube_8b8b.txt"
pattern = r"runtime (\d+(\.\d+)?) ms"

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

writer = open('0_spmm_magicube_8b8b.txt', 'w')
for r in tops:
    writer.write(str(r) + '\n')
writer.close()



filename = "../SpMM/SpMM/spmm_magicube_8b4b.txt"
pattern = r"runtime (\d+(\.\d+)?) ms"

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

writer = open('0_spmm_magicube_8b4b.txt', 'w')
for r in tops:
    writer.write(str(r) + '\n')
writer.close()



filename = "../SpMM/SpMM/spmm_magicube_4b4b.txt"
pattern = r"runtime (\d+(\.\d+)?) ms"

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

writer = open('0_spmm_magicube_4b4b.txt', 'w')
for r in tops:
    writer.write(str(r) + '\n')
writer.close()
