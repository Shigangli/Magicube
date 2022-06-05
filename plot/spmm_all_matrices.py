import re
import six


#filename = "0_4b4b_spmm_all_matrices.txt"
#pattern = r"Runtime (\d+(\.\d+)?) ms"
#
#reader = open(filename, 'r')
#runtime = []
#
#while True:
#    line = reader.readline()
#    if len(line) == 0:
#        break
#    line = line.rstrip()
#    m = re.search(pattern, line)
#    if m:
#        runtime.append(m.group(1))
#
#writer = open('1-4b4b.txt', 'w')
#for r in runtime:
#    writer.write(str(r) + '\n')
#writer.close()
#
#
#
#
#
#filename = "0_8b4b_spmm_all_matrices.txt"
#pattern = r"Runtime (\d+(\.\d+)?) ms"
#
#reader = open(filename, 'r')
#runtime = []
#
#while True:
#    line = reader.readline()
#    if len(line) == 0:
#        break
#    line = line.rstrip()
#    m = re.search(pattern, line)
#    if m:
#        runtime.append(m.group(1))
#
#writer = open('1-8b4b.txt', 'w')
#for r in runtime:
#    writer.write(str(r) + '\n')
#writer.close()
#
#
#
#
#filename = "0_8b8b_spmm_all_matrices.txt"
#pattern = r"Runtime (\d+(\.\d+)?) ms"
#
#reader = open(filename, 'r')
#runtime = []
#
#while True:
#    line = reader.readline()
#    if len(line) == 0:
#        break
#    line = line.rstrip()
#    m = re.search(pattern, line)
#    if m:
#        runtime.append(m.group(1))
#
#writer = open('1-8b8b.txt', 'w')
#for r in runtime:
#    writer.write(str(r) + '\n')
#writer.close()
#
#
#
#
#filename = "0_16b8b_spmm_all_matrices.txt"
#pattern = r"Runtime (\d+(\.\d+)?) ms"
#
#reader = open(filename, 'r')
#runtime = []
#
#while True:
#    line = reader.readline()
#    if len(line) == 0:
#        break
#    line = line.rstrip()
#    m = re.search(pattern, line)
#    if m:
#        runtime.append(m.group(1))
#
#writer = open('1-16b8b.txt', 'w')
#for r in runtime:
#    writer.write(str(r) + '\n')
#writer.close()
#
#
#
#filename = "sc21_spmm.txt"
#pattern = r"runtime (\d+(\.\d+)?) ms"
#
#reader = open(filename, 'r')
#runtime = []
#
#while True:
#    line = reader.readline()
#    if len(line) == 0:
#        break
#    line = line.rstrip()
#    m = re.search(pattern, line)
#    if m:
#        runtime.append(m.group(1))
#
#writer = open('1-vectorSparse.txt', 'w')
#for r in runtime:
#    writer.write(str(r) + '\n')
#writer.close()
#
#
#
#
#filename = "cusparse_f16_bell.txt"
#pattern = r"runtime (\d+(\.\d+)?) ms"
#
#reader = open(filename, 'r')
#runtime = []
#
#while True:
#    line = reader.readline()
#    if len(line) == 0:
#        break
#    line = line.rstrip()
#    m = re.search(pattern, line)
#    if m:
#        runtime.append(m.group(1))
#
#writer = open('1-cusparse_f16_bell.txt', 'w')
#for r in runtime:
#    writer.write(str(r) + '\n')
#writer.close()



filename = "cusparse_int8_bell.txt"
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

writer = open('2-cusparse_int8_bell.txt', 'w')
for r in runtime:
    writer.write(str(r) + '\n')
writer.close()



#filename = "cublas_f16_dense.txt"
#pattern = r"runtime (\d+(\.\d+)?) ms"
#
#reader = open(filename, 'r')
#runtime = []
#
#while True:
#    line = reader.readline()
#    if len(line) == 0:
#        break
#    line = line.rstrip()
#    m = re.search(pattern, line)
#    if m:
#        runtime.append(m.group(1))
#
#writer = open('1-cublas_f16_dense.txt', 'w')
#for r in runtime:
#    writer.write(str(r) + '\n')
#writer.close()

filename = "cublas_int8_dense.txt"
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

writer = open('2-cublas_int8_dense.txt', 'w')
for r in runtime:
    writer.write(str(r) + '\n')
writer.close()

#0_16b8b_spmm_all_matrices.txt  0_8b4b_spmm_all_matrices.txt  cusparse_f16_bell.txt 
#0_4b4b_spmm_all_matrices.txt   0_8b8b_spmm_all_matrices.txt  cublas_f16_dense.txt  sc21_spmm.txt

