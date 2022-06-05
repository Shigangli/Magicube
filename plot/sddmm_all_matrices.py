import re
import six


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

writer = open('0_sddmm_cublas_fp16.txt', 'w')
for r in runtime:
    writer.write(str(r) + '\n')
writer.close()


filename = "../baselines/sddmm_cublas_int8.txt"
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

writer = open('0_sddmm_cublas_int8.txt', 'w')
for r in runtime:
    writer.write(str(r) + '\n')
writer.close()


filename = "../baselines/sddmm_vectorSparse.txt"
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

writer = open('0_sddmm_vectorSparse.txt', 'w')
for r in runtime:
    writer.write(str(r) + '\n')
writer.close()


filename = "../SDDMM/SDDMM/sddmm_magicube_16b16b.txt"
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

writer = open('0_sddmm_magicube_16b16b.txt', 'w')
for r in runtime:
    writer.write(str(r) + '\n')
writer.close()


filename = "../SDDMM/SDDMM/sddmm_magicube_8b8b.txt"
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

writer = open('0_sddmm_magicube_8b8b.txt', 'w')
for r in runtime:
    writer.write(str(r) + '\n')
writer.close()


filename = "../SDDMM/SDDMM/sddmm_magicube_4b4b.txt"
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

writer = open('0_sddmm_magicube_4b4b.txt', 'w')
for r in runtime:
    writer.write(str(r) + '\n')
writer.close()
