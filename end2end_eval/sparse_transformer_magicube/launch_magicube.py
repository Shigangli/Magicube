import os

for i in range(256):
    cmd = 'python3 end_to_end.py --model sparse --bs 2 --lhs_pre 4 --rhs_pre 4'
    os.system(cmd)

for i in range(256):
    cmd = 'python3 end_to_end.py --model sparse --bs 2 --lhs_pre 8 --rhs_pre 8'
    os.system(cmd)

for i in range(256):
    cmd = 'python3 end_to_end.py --model sparse --bs 2 --lhs_pre 16 --rhs_pre 8'
    os.system(cmd)

for i in range(256):
    cmd = 'python3 end_to_end.py --model sparse --bs 8 --lhs_pre 4 --rhs_pre 4'
    os.system(cmd)

for i in range(256):
    cmd = 'python3 end_to_end.py --model sparse --bs 8 --lhs_pre 8 --rhs_pre 8'
    os.system(cmd)

for i in range(256):
    cmd = 'python3 end_to_end.py --model sparse --bs 8 --lhs_pre 16 --rhs_pre 8'
    os.system(cmd)
