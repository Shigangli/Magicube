import re
import six

filename = "0-sddmm-cublase-int8-dense.txt"
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

writer = open('1-sddmm-cublase-int8-dense.txt', 'w')
for r in runtime:
    writer.write(str(r) + '\n')
writer.close()


filename = "0-sddmm-cublase-f16-dense.txt"
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

writer = open('1-sddmm-cublase-f16-dense.txt', 'w')
for r in runtime:
    writer.write(str(r) + '\n')
writer.close()



filename = "0-sc21-sddmm.txt"
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

writer = open('1-sc21-sddmm.txt', 'w')
for r in runtime:
    writer.write(str(r) + '\n')
writer.close()



filename = "0-magicube-sddmm-4b4b.txt"
pattern = r"Runtime: (\d+(\.\d+)?) ms"

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

writer = open('1-sddmm-4b4b.txt', 'w')
for r in runtime:
    writer.write(str(r) + '\n')
writer.close()





filename = "0-magicube-sddmm-8b8b.txt"
pattern = r"Runtime: (\d+(\.\d+)?) ms"

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

writer = open('1-sddmm-8b8b.txt', 'w')
for r in runtime:
    writer.write(str(r) + '\n')
writer.close()




filename = "0-magicube-sddmm-16b16b.txt"
pattern = r"Runtime: (\d+(\.\d+)?) ms"

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

writer = open('1-sddmm-16b16b.txt', 'w')
for r in runtime:
    writer.write(str(r) + '\n')
writer.close()




