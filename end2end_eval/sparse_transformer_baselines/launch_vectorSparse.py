import os

for i in range(8):
    cmd = 'python3 end_to_end.py --model sparse --sparsity 0.9 --mlp_dim 1024 --embed_dim 256 --num_heads 4 --seq_len 4096 --bs 2'
    os.system(cmd)
print(cmd, flush=True)

for i in range(8):
    cmd = 'python3 end_to_end.py --model sparse --sparsity 0.9 --mlp_dim 1024 --embed_dim 256 --num_heads 4 --seq_len 4096 --bs 8'
    os.system(cmd)
print(cmd, flush=True)

for i in range(8):
    cmd = 'python3 end_to_end.py --model sparse --sparsity 0.9 --mlp_dim 2048 --embed_dim 512 --num_heads 8 --seq_len 4096 --bs 2'
    os.system(cmd)
print(cmd, flush=True)

for i in range(8):
    cmd = 'python3 end_to_end.py --model sparse --sparsity 0.9 --mlp_dim 2048 --embed_dim 512 --num_heads 8 --seq_len 4096 --bs 8'
    os.system(cmd)
print(cmd, flush=True)

for i in range(8):
    cmd = 'python3 end_to_end.py --model sparse --sparsity 0.9 --mlp_dim 1024 --embed_dim 256 --num_heads 4 --seq_len 8192 --bs 2'
    os.system(cmd)
print(cmd, flush=True)

for i in range(8):
    cmd = 'python3 end_to_end.py --model sparse --sparsity 0.9 --mlp_dim 1024 --embed_dim 256 --num_heads 4 --seq_len 8192 --bs 8'
    os.system(cmd)
print(cmd, flush=True)

for i in range(8):
    cmd = 'python3 end_to_end.py --model sparse --sparsity 0.9 --mlp_dim 2048 --embed_dim 512 --num_heads 8 --seq_len 8192 --bs 2'
    os.system(cmd)
print(cmd, flush=True)

for i in range(8):
    cmd = 'python3 end_to_end.py --model sparse --sparsity 0.9 --mlp_dim 2048 --embed_dim 512 --num_heads 8 --seq_len 8192 --bs 8'
    os.system(cmd)
print(cmd, flush=True)


for i in range(8):
    cmd = 'python3 end_to_end.py --model sparse --sparsity 0.95 --mlp_dim 1024 --embed_dim 256 --num_heads 4 --seq_len 4096 --bs 2'
    os.system(cmd)
print(cmd, flush=True)

for i in range(8):
    cmd = 'python3 end_to_end.py --model sparse --sparsity 0.95 --mlp_dim 1024 --embed_dim 256 --num_heads 4 --seq_len 4096 --bs 8'
    os.system(cmd)
print(cmd, flush=True)

for i in range(8):
    cmd = 'python3 end_to_end.py --model sparse --sparsity 0.95 --mlp_dim 2048 --embed_dim 512 --num_heads 8 --seq_len 4096 --bs 2'
    os.system(cmd)
print(cmd, flush=True)

for i in range(8):
    cmd = 'python3 end_to_end.py --model sparse --sparsity 0.95 --mlp_dim 2048 --embed_dim 512 --num_heads 8 --seq_len 4096 --bs 8'
    os.system(cmd)
print(cmd, flush=True)

for i in range(8):
    cmd = 'python3 end_to_end.py --model sparse --sparsity 0.95 --mlp_dim 1024 --embed_dim 256 --num_heads 4 --seq_len 8192 --bs 2'
    os.system(cmd)
print(cmd, flush=True)

for i in range(8):
    cmd = 'python3 end_to_end.py --model sparse --sparsity 0.95 --mlp_dim 1024 --embed_dim 256 --num_heads 4 --seq_len 8192 --bs 8'
    os.system(cmd)
print(cmd, flush=True)

for i in range(8):
    cmd = 'python3 end_to_end.py --model sparse --sparsity 0.95 --mlp_dim 2048 --embed_dim 512 --num_heads 8 --seq_len 8192 --bs 2'
    os.system(cmd)
print(cmd, flush=True)

for i in range(8):
    cmd = 'python3 end_to_end.py --model sparse --sparsity 0.95 --mlp_dim 2048 --embed_dim 512 --num_heads 8 --seq_len 8192 --bs 8'
    os.system(cmd)
print(cmd, flush=True)
