./spmm_benchmark  /users/shigang/gitrepo/dlmc/rn50/random_pruning/0.7/bottleneck_2_block_group3_5_1.smtx 512 8 0 0 1 1 1
./spmm_benchmark  /users/shigang/gitrepo/dlmc/rn50/random_pruning/0.7/bottleneck_2_block_group3_5_1.smtx 512 4 0 1 1 1 8 8
./spmm_benchmark  /users/shigang/gitrepo/dlmc/rn50/random_pruning/0.7/bottleneck_2_block_group3_5_1.smtx 512 4 0 1 1 1 4 4
./spmm_benchmark  /users/shigang/gitrepo/dlmc/rn50/random_pruning/0.7/bottleneck_2_block_group3_5_1.smtx 512 8 0 1 1 1 4 4
CUDA_VISIBLE_DEVICES=GPU-31acddbe-f963-b876-2508-0c529c73da36 ./spmm_benchmark  /users/shigang/gitrepo/dlmc/rn50/random_pruning/0.7/bottleneck_2_block_group3_5_1.smtx 512 8 0 1 1 1 4 4
nsys profile --force-overwrite true  -t cuda -o spmm_report ./spmm_benchmark  /users/shigang/gitrepo/dlmc/rn50/random_pruning/0.7/bottleneck_2_block_group3_5_1.smtx 512 8 0 1 1 1 8 4
