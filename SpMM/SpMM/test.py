import os
import argparse
import numpy as np

#cmd = 'python3 ncu_profile.py --bm %s -k %d -v %d --kernel %s --prof --job sddmm --sddmm_alg %s --precision %s' % (benchmark, args.dimK, args.dimV, args.kernel, args.sddmm_alg, args.precision)
cmd = './spmm_benchmark  /users/shigang/gitrepo/dlmc/rn50/random_pruning/0.7/bottleneck_2_block_group3_5_1.smtx 512 8 0 1 1 1 4 4'
os.system(cmd)
