import os

for i in range(256):
    cmd = 'python3 end_to_end.py --model sparse --bs 2'
    os.system(cmd)

for i in range(256):
    cmd = 'python3 end_to_end.py --model sparse --bs 8'
    os.system(cmd)
