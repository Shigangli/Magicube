import re
import six


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

writer = open('0_pytorch_n2n.txt', 'w')
for r in runtime:
    writer.write(str(r) + '\n')
writer.close()


filename = "../end2end_eval/sparse_transformer_baselines/vectorSparse_n2n.txt"
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

writer = open('0_vectorSparse_n2n.txt', 'w')
for r in runtime:
    writer.write(str(r) + '\n')
writer.close()


filename = "../end2end_eval/sparse_transformer_magicube/magicube_n2n.txt"
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

writer = open('0_magicube_n2n.txt', 'w')
for r in runtime:
    writer.write(str(r) + '\n')
writer.close()
