import os
import argparse

# Args
parser = argparse.ArgumentParser(description='launch the experiments')

parser.add_argument('--exp', choices=['sddmm', 'spmm', 'reuse', 'bell'], help='the experiment to run')

args = parser.parse_args()

if args.exp == 'spmm':
    for k in [64, 128, 256]:
        cmd = 'python3 job_launcher.py --start 0 --end 323 --dimK %d --dimV 1 --kernel dense --job spmm --sort --precision half' % k
        os.system(cmd)

        cmd = 'python3 job_launcher.py --start 0 --end 323 --dimK %d --dimV 1 --kernel sputnik --job spmm --sort --precision half' % k
        os.system(cmd)

        Vs = [2, 4, 8]

        for v in Vs:
            cmd = 'python3 job_launcher.py --start 0 --end 323 --dimK %d --dimV %d --kernel dense --job spmm --sort --precision half' % (k, v)
            os.system(cmd)

            cmd = 'python3 job_launcher.py --start 0 --end 323 --dimK %d --dimV %d --kernel cuda --job spmm --sort --precision half' % (k, v)
            os.system(cmd)

            cmd = 'python3 job_launcher.py --start 0 --end 323 --dimK %d --dimV %d --kernel wmma --job spmm --sort --precision half' % (k, v)
            os.system(cmd)

            cmd = 'python3 job_launcher.py --start 0 --end 323 --dimK %d --dimV %d --kernel bell --job spmm --sort --precision half' % (k, v)
            os.system(cmd)
    
    # plot
    cmd = 'python3 plot_spmm.py --start 0 --end 323 --dimK 256 --sort --bm rn50 --combo'
    os.system(cmd)

elif args.exp == 'sddmm':
    for k in [64, 128, 256]:
        cmd = 'python3 job_launcher.py --start 0 --end 323 --dimK %d --dimV 1 --kernel dense --sort --job sddmm --precision half' % k
        os.system(cmd)

        cmd = 'python3 job_launcher.py --start 0 --end 323 --dimK %d --dimV 1 --kernel cuda --sort  --job sddmm --precision half' % k
        os.system(cmd)

        Vs = [2, 4, 8]

        for v in Vs:
            cmd = 'python3 job_launcher.py --start 0 --end 323 --dimK %d --dimV %d --kernel dense --sort --job sddmm --precision half' % (k, v)
            os.system(cmd)

            cmd = 'python3 job_launcher.py --start 0 --end 323 --dimK %d --dimV %d --kernel cuda --sort --job sddmm --precision half' % (k, v)
            os.system(cmd)

            cmd = 'python3 job_launcher.py --start 0 --end 323 --dimK %d --dimV %d --kernel wmma --sort --job sddmm --sddmm_alg wmma --precision half' % (k, v)
            os.system(cmd)

            cmd = 'python3 job_launcher.py --start 0 --end 323 --dimK %d --dimV %d --kernel wmma --sort --job sddmm --sddmm_alg mma_reg --precision half' % (k, v)
            os.system(cmd)

            cmd = 'python3 job_launcher.py --start 0 --end 323 --dimK %d --dimV %d --kernel wmma --sort --job sddmm --sddmm_alg mma_shfl --precision half' % (k, v)
            os.system(cmd)

            cmd = 'python3 job_launcher.py --start 0 --end 323 --dimK %d --dimV %d --kernel wmma --sort --job sddmm --sddmm_alg mma_arch --precision half' % (k, v)
            os.system(cmd)
    
    # plot
    cmd = 'python3 plot_sddmm.py --start 0 --end 323 --dimK 256 --sort --bm rn50 --combo'
    os.system(cmd)

elif args.exp == 'reuse':
    for sparse in ['0.5', '0.7', '0.8', '0.9', '0.95', '0.98']:
        cmd = 'python3 ncu_profile.py --bm /users/shigang/gitrepo/dlmc/rn50/magnitude_pruning/0.5/bottleneck_projection_block_group_projection_block_group4.smtx -k 256 -v 2 --kernel bell --prof --job spmm --precision half --mem'
        os.system(cmd)
        cmd = 'python3 ncu_profile.py --bm /users/shigang/gitrepo/dlmc/rn50/magnitude_pruning/0.5/bottleneck_projection_block_group_projection_block_group4.smtx -k 256 -v 2 --kernel wmma --sort --prof --job spmm --precision half --mem'
        os.system(cmd)
        cmd = 'python3 ncu_profile.py --bm /users/shigang/gitrepo/dlmc/rn50/magnitude_pruning/0.5/bottleneck_projection_block_group_projection_block_group4.smtx -k 256 -v 4 --kernel bell --prof --job spmm --precision half --mem'
        os.system(cmd)
        cmd = 'python3 ncu_profile.py --bm /users/shigang/gitrepo/dlmc/rn50/magnitude_pruning/0.5/bottleneck_projection_block_group_projection_block_group4.smtx -k 256 -v 4 --kernel wmma --sort --prof --job spmm --precision half --mem'
        os.system(cmd)
        cmd = 'python3 ncu_profile.py --bm /users/shigang/gitrepo/dlmc/rn50/magnitude_pruning/0.5/bottleneck_projection_block_group_projection_block_group4.smtx -k 256 -v 8 --kernel bell --prof --job spmm --precision half --mem'
        os.system(cmd)
        cmd = 'python3 ncu_profile.py --bm /users/shigang/gitrepo/dlmc/rn50/magnitude_pruning/0.5/bottleneck_projection_block_group_projection_block_group4.smtx -k 256 -v 8 --kernel wmma --sort --prof --job spmm --precision half --mem'
        os.system(cmd)

        cmd = 'python3 ncu_profile.py --bm /users/shigang/gitrepo/dlmc/rn50/magnitude_pruning/0.7/bottleneck_projection_block_group_projection_block_group4.smtx -k 256 -v 2 --kernel bell --prof --job spmm --precision half --mem'
        os.system(cmd)
        cmd = 'python3 ncu_profile.py --bm /users/shigang/gitrepo/dlmc/rn50/magnitude_pruning/0.7/bottleneck_projection_block_group_projection_block_group4.smtx -k 256 -v 2 --kernel wmma --sort --prof --job spmm --precision half --mem'
        os.system(cmd)
        cmd = 'python3 ncu_profile.py --bm /users/shigang/gitrepo/dlmc/rn50/magnitude_pruning/0.7/bottleneck_projection_block_group_projection_block_group4.smtx -k 256 -v 4 --kernel bell --prof --job spmm --precision half --mem'
        os.system(cmd)
        cmd = 'python3 ncu_profile.py --bm /users/shigang/gitrepo/dlmc/rn50/magnitude_pruning/0.7/bottleneck_projection_block_group_projection_block_group4.smtx -k 256 -v 4 --kernel wmma --sort --prof --job spmm --precision half --mem'
        os.system(cmd)
        cmd = 'python3 ncu_profile.py --bm /users/shigang/gitrepo/dlmc/rn50/magnitude_pruning/0.7/bottleneck_projection_block_group_projection_block_group4.smtx -k 256 -v 8 --kernel bell --prof --job spmm --precision half --mem'
        os.system(cmd)
        cmd = 'python3 ncu_profile.py --bm /users/shigang/gitrepo/dlmc/rn50/magnitude_pruning/0.7/bottleneck_projection_block_group_projection_block_group4.smtx -k 256 -v 8 --kernel wmma --sort --prof --job spmm --precision half --mem'
        os.system(cmd)

        cmd = 'python3 ncu_profile.py --bm /users/shigang/gitrepo/dlmc/rn50/magnitude_pruning/0.8/bottleneck_projection_block_group_projection_block_group4.smtx -k 256 -v 2 --kernel bell --prof --job spmm --precision half --mem'
        os.system(cmd)
        cmd = 'python3 ncu_profile.py --bm /users/shigang/gitrepo/dlmc/rn50/magnitude_pruning/0.8/bottleneck_projection_block_group_projection_block_group4.smtx -k 256 -v 2 --kernel wmma --sort --prof --job spmm --precision half --mem'
        os.system(cmd)
        cmd = 'python3 ncu_profile.py --bm /users/shigang/gitrepo/dlmc/rn50/magnitude_pruning/0.8/bottleneck_projection_block_group_projection_block_group4.smtx -k 256 -v 4 --kernel bell --prof --job spmm --precision half --mem'
        os.system(cmd)
        cmd = 'python3 ncu_profile.py --bm /users/shigang/gitrepo/dlmc/rn50/magnitude_pruning/0.8/bottleneck_projection_block_group_projection_block_group4.smtx -k 256 -v 4 --kernel wmma --sort --prof --job spmm --precision half --mem'
        os.system(cmd)
        cmd = 'python3 ncu_profile.py --bm /users/shigang/gitrepo/dlmc/rn50/magnitude_pruning/0.8/bottleneck_projection_block_group_projection_block_group4.smtx -k 256 -v 8 --kernel bell --prof --job spmm --precision half --mem'
        os.system(cmd)
        cmd = 'python3 ncu_profile.py --bm /users/shigang/gitrepo/dlmc/rn50/magnitude_pruning/0.8/bottleneck_projection_block_group_projection_block_group4.smtx -k 256 -v 8 --kernel wmma --sort --prof --job spmm --precision half --mem'
        os.system(cmd)

        cmd = 'python3 ncu_profile.py --bm /users/shigang/gitrepo/dlmc/rn50/magnitude_pruning/0.9/bottleneck_projection_block_group_projection_block_group4.smtx -k 256 -v 2 --kernel bell --prof --job spmm --precision half --mem'
        os.system(cmd)
        cmd = 'python3 ncu_profile.py --bm /users/shigang/gitrepo/dlmc/rn50/magnitude_pruning/0.9/bottleneck_projection_block_group_projection_block_group4.smtx -k 256 -v 2 --kernel wmma --sort --prof --job spmm --precision half --mem'
        os.system(cmd)
        cmd = 'python3 ncu_profile.py --bm /users/shigang/gitrepo/dlmc/rn50/magnitude_pruning/0.9/bottleneck_projection_block_group_projection_block_group4.smtx -k 256 -v 4 --kernel bell --prof --job spmm --precision half --mem'
        os.system(cmd)
        cmd = 'python3 ncu_profile.py --bm /users/shigang/gitrepo/dlmc/rn50/magnitude_pruning/0.9/bottleneck_projection_block_group_projection_block_group4.smtx -k 256 -v 4 --kernel wmma --sort --prof --job spmm --precision half --mem'
        os.system(cmd)
        cmd = 'python3 ncu_profile.py --bm /users/shigang/gitrepo/dlmc/rn50/magnitude_pruning/0.9/bottleneck_projection_block_group_projection_block_group4.smtx -k 256 -v 8 --kernel bell --prof --job spmm --precision half --mem'
        os.system(cmd)
        cmd = 'python3 ncu_profile.py --bm /users/shigang/gitrepo/dlmc/rn50/magnitude_pruning/0.9/bottleneck_projection_block_group_projection_block_group4.smtx -k 256 -v 8 --kernel wmma --sort --prof --job spmm --precision half --mem'
        os.system(cmd)

        cmd = 'python3 ncu_profile.py --bm /users/shigang/gitrepo/dlmc/rn50/magnitude_pruning/0.95/bottleneck_projection_block_group_projection_block_group4.smtx -k 256 -v 2 --kernel bell --prof --job spmm --precision half --mem'
        os.system(cmd)
        cmd = 'python3 ncu_profile.py --bm /users/shigang/gitrepo/dlmc/rn50/magnitude_pruning/0.95/bottleneck_projection_block_group_projection_block_group4.smtx -k 256 -v 2 --kernel wmma --sort --prof --job spmm --precision half --mem'
        os.system(cmd)
        cmd = 'python3 ncu_profile.py --bm /users/shigang/gitrepo/dlmc/rn50/magnitude_pruning/0.95/bottleneck_projection_block_group_projection_block_group4.smtx -k 256 -v 4 --kernel bell --prof --job spmm --precision half --mem'
        os.system(cmd)
        cmd = 'python3 ncu_profile.py --bm /users/shigang/gitrepo/dlmc/rn50/magnitude_pruning/0.95/bottleneck_projection_block_group_projection_block_group4.smtx -k 256 -v 4 --kernel wmma --sort --prof --job spmm --precision half --mem'
        os.system(cmd)
        cmd = 'python3 ncu_profile.py --bm /users/shigang/gitrepo/dlmc/rn50/magnitude_pruning/0.95/bottleneck_projection_block_group_projection_block_group4.smtx -k 256 -v 8 --kernel bell --prof --job spmm --precision half --mem'
        os.system(cmd)
        cmd = 'python3 ncu_profile.py --bm /users/shigang/gitrepo/dlmc/rn50/magnitude_pruning/0.95/bottleneck_projection_block_group_projection_block_group4.smtx -k 256 -v 8 --kernel wmma --sort --prof --job spmm --precision half --mem'
        os.system(cmd)

        cmd = 'python3 ncu_profile.py --bm /users/shigang/gitrepo/dlmc/rn50/magnitude_pruning/0.98/bottleneck_projection_block_group_projection_block_group4.smtx -k 256 -v 2 --kernel bell --prof --job spmm --precision half --mem'
        os.system(cmd)
        cmd = 'python3 ncu_profile.py --bm /users/shigang/gitrepo/dlmc/rn50/magnitude_pruning/0.98/bottleneck_projection_block_group_projection_block_group4.smtx -k 256 -v 2 --kernel wmma --sort --prof --job spmm --precision half --mem'
        os.system(cmd)
        cmd = 'python3 ncu_profile.py --bm /users/shigang/gitrepo/dlmc/rn50/magnitude_pruning/0.98/bottleneck_projection_block_group_projection_block_group4.smtx -k 256 -v 4 --kernel bell --prof --job spmm --precision half --mem'
        os.system(cmd)
        cmd = 'python3 ncu_profile.py --bm /users/shigang/gitrepo/dlmc/rn50/magnitude_pruning/0.98/bottleneck_projection_block_group_projection_block_group4.smtx -k 256 -v 4 --kernel wmma --sort --prof --job spmm --precision half --mem'
        os.system(cmd)
        cmd = 'python3 ncu_profile.py --bm /users/shigang/gitrepo/dlmc/rn50/magnitude_pruning/0.98/bottleneck_projection_block_group_projection_block_group4.smtx -k 256 -v 8 --kernel bell --prof --job spmm --precision half --mem'
        os.system(cmd)
        cmd = 'python3 ncu_profile.py --bm /users/shigang/gitrepo/dlmc/rn50/magnitude_pruning/0.98/bottleneck_projection_block_group_projection_block_group4.smtx -k 256 -v 8 --kernel wmma --sort --prof --job spmm --precision half --mem'
        os.system(cmd)
elif args.exp == 'bell':
    cmd = 'python3 job_launcher.py --start 0 --end 323 --dimK 256 --dimV 2 --kernel dense --job spmm --precision half'
    os.system(cmd)

    cmd = 'python3 job_launcher.py --start 0 --end 323 --dimK 256 --dimV 4 --kernel dense --job spmm --precision half'
    os.system(cmd)

    cmd = 'python3 job_launcher.py --start 0 --end 323 --dimK 256 --dimV 8 --kernel dense --job spmm --precision half'
    os.system(cmd)

    cmd = 'python3 job_launcher.py --start 0 --end 323 --dimK 256 --dimV 16 --kernel dense --job spmm --precision half'
    os.system(cmd)

    cmd = 'python3 job_launcher.py --start 0 --end 323 --dimK 256 --dimV 2 --kernel bell --job spmm --precision half'
    os.system(cmd)

    cmd = 'python3 job_launcher.py --start 0 --end 323 --dimK 256 --dimV 4 --kernel bell --job spmm --precision half'
    os.system(cmd)

    cmd = 'python3 job_launcher.py --start 0 --end 323 --dimK 256 --dimV 8 --kernel bell --job spmm --precision half'
    os.system(cmd)

    cmd = 'python3 job_launcher.py --start 0 --end 323 --dimK 256 --dimV 16 --kernel bell --job spmm --precision half'
    os.system(cmd)
elif args.exp == 'finegrained':
    cmd = 'python3 job_launcher.py --start 0 --end 323 --dimK 256 --dimV 1 --kernel dense --job sddmm --precision half'
    os.system(cmd)

    cmd = 'python3 job_launcher.py --start 0 --end 323 --dimK 256 --dimV 1 --kernel sputnik --job sddmm --precision half'
    os.system(cmd)

    cmd = 'python3 job_launcher.py --start 0 --end 323 --dimK 256 --dimV 1 --kernel dense --job sddmm --precision single'
    os.system(cmd)

    cmd = 'python3 job_launcher.py --start 0 --end 323 --dimK 256 --dimV 1 --kernel sputnik --job sddmm --precision single'
    os.system(cmd)

    cmd = 'python3 job_launcher.py --start 0 --end 323 --dimK 256 --dimV 1 --kernel cusparse --job sddmm --precision single'
    os.system(cmd)

    cmd = 'python3 job_launcher.py --start 0 --end 323 --dimK 256 --dimV 1 --kernel dense --sort --job spmm --precision half'
    os.system(cmd)

    cmd = 'python3 job_launcher.py --start 0 --end 323 --dimK 256 --dimV 1 --kernel sputnik --sort --job spmm --precision half'
    os.system(cmd)

    cmd = 'python3 job_launcher.py --start 0 --end 323 --dimK 256 --dimV 1 --kernel cusparse --sort --job spmm --precision half'
    os.system(cmd)

    cmd = 'python3 job_launcher.py --start 0 --end 323 --dimK 256 --dimV 1 --kernel dense --sort --job spmm --precision single'
    os.system(cmd)

    cmd = 'python3 job_launcher.py --start 0 --end 323 --dimK 256 --dimV 1 --kernel sputnik --sort --job spmm --precision single'
    os.system(cmd)

    cmd = 'python3 job_launcher.py --start 0 --end 323 --dimK 256 --dimV 1 --kernel cusparse --sort --job spmm --precision single'
    os.system(cmd)
else:
    print('unrecognized expriment')
