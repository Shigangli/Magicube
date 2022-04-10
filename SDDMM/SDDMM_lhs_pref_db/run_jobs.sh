echo -e "Evaluation perf for different precisions: K = 1024, Iteration = 1024 \n"

echo -e "L16-R16 \n"
./sddmm_benchmark  /users/shigang/gitrepo/dlmc/rn50/random_pruning/0.5/bottleneck_2_block_group3_5_1.smtx 1024 2 1 1 1 16 16
echo -e "\n"
./sddmm_benchmark  /users/shigang/gitrepo/dlmc/rn50/random_pruning/0.5/bottleneck_2_block_group3_5_1.smtx 1024 4 1 1 1 16 16
echo -e "\n"
./sddmm_benchmark  /users/shigang/gitrepo/dlmc/rn50/random_pruning/0.5/bottleneck_2_block_group3_5_1.smtx 1024 8 1 1 1 16 16
echo -e "\n"
./sddmm_benchmark  /users/shigang/gitrepo/dlmc/rn50/random_pruning/0.7/bottleneck_2_block_group3_5_1.smtx 1024 2 1 1 1 16 16
echo -e "\n"
./sddmm_benchmark  /users/shigang/gitrepo/dlmc/rn50/random_pruning/0.7/bottleneck_2_block_group3_5_1.smtx 1024 4 1 1 1 16 16
echo -e "\n"
./sddmm_benchmark  /users/shigang/gitrepo/dlmc/rn50/random_pruning/0.7/bottleneck_2_block_group3_5_1.smtx 1024 8 1 1 1 16 16
echo -e "\n"
./sddmm_benchmark  /users/shigang/gitrepo/dlmc/rn50/random_pruning/0.8/bottleneck_2_block_group3_5_1.smtx 1024 2 1 1 1 16 16
echo -e "\n"
./sddmm_benchmark  /users/shigang/gitrepo/dlmc/rn50/random_pruning/0.8/bottleneck_2_block_group3_5_1.smtx 1024 4 1 1 1 16 16
echo -e "\n"
./sddmm_benchmark  /users/shigang/gitrepo/dlmc/rn50/random_pruning/0.8/bottleneck_2_block_group3_5_1.smtx 1024 8 1 1 1 16 16
echo -e "\n"
./sddmm_benchmark  /users/shigang/gitrepo/dlmc/rn50/random_pruning/0.9/bottleneck_2_block_group3_5_1.smtx 1024 2 1 1 1 16 16
echo -e "\n"
./sddmm_benchmark  /users/shigang/gitrepo/dlmc/rn50/random_pruning/0.9/bottleneck_2_block_group3_5_1.smtx 1024 4 1 1 1 16 16
echo -e "\n"
./sddmm_benchmark  /users/shigang/gitrepo/dlmc/rn50/random_pruning/0.9/bottleneck_2_block_group3_5_1.smtx 1024 8 1 1 1 16 16
echo -e "\n"
./sddmm_benchmark  /users/shigang/gitrepo/dlmc/rn50/random_pruning/0.95/bottleneck_2_block_group3_5_1.smtx 1024 2 1 1 1 16 16
echo -e "\n"
./sddmm_benchmark  /users/shigang/gitrepo/dlmc/rn50/random_pruning/0.95/bottleneck_2_block_group3_5_1.smtx 1024 4 1 1 1 16 16
echo -e "\n"
./sddmm_benchmark  /users/shigang/gitrepo/dlmc/rn50/random_pruning/0.95/bottleneck_2_block_group3_5_1.smtx 1024 8 1 1 1 16 16
echo -e "\n"
./sddmm_benchmark  /users/shigang/gitrepo/dlmc/rn50/random_pruning/0.98/bottleneck_2_block_group3_5_1.smtx 1024 2 1 1 1 16 16
echo -e "\n"
./sddmm_benchmark  /users/shigang/gitrepo/dlmc/rn50/random_pruning/0.98/bottleneck_2_block_group3_5_1.smtx 1024 4 1 1 1 16 16
echo -e "\n"
./sddmm_benchmark  /users/shigang/gitrepo/dlmc/rn50/random_pruning/0.98/bottleneck_2_block_group3_5_1.smtx 1024 8 1 1 1 16 16
echo -e "\n"


echo -e "L8-R8 \n"
./sddmm_benchmark  /users/shigang/gitrepo/dlmc/rn50/random_pruning/0.5/bottleneck_2_block_group3_5_1.smtx 1024 2 1 1 1 8 8
echo -e "\n"
./sddmm_benchmark  /users/shigang/gitrepo/dlmc/rn50/random_pruning/0.5/bottleneck_2_block_group3_5_1.smtx 1024 4 1 1 1 8 8
echo -e "\n"
./sddmm_benchmark  /users/shigang/gitrepo/dlmc/rn50/random_pruning/0.5/bottleneck_2_block_group3_5_1.smtx 1024 8 1 1 1 8 8
echo -e "\n"
./sddmm_benchmark  /users/shigang/gitrepo/dlmc/rn50/random_pruning/0.7/bottleneck_2_block_group3_5_1.smtx 1024 2 1 1 1 8 8
echo -e "\n"
./sddmm_benchmark  /users/shigang/gitrepo/dlmc/rn50/random_pruning/0.7/bottleneck_2_block_group3_5_1.smtx 1024 4 1 1 1 8 8
echo -e "\n"
./sddmm_benchmark  /users/shigang/gitrepo/dlmc/rn50/random_pruning/0.7/bottleneck_2_block_group3_5_1.smtx 1024 8 1 1 1 8 8
echo -e "\n"
./sddmm_benchmark  /users/shigang/gitrepo/dlmc/rn50/random_pruning/0.8/bottleneck_2_block_group3_5_1.smtx 1024 2 1 1 1 8 8
echo -e "\n"
./sddmm_benchmark  /users/shigang/gitrepo/dlmc/rn50/random_pruning/0.8/bottleneck_2_block_group3_5_1.smtx 1024 4 1 1 1 8 8
echo -e "\n"
./sddmm_benchmark  /users/shigang/gitrepo/dlmc/rn50/random_pruning/0.8/bottleneck_2_block_group3_5_1.smtx 1024 8 1 1 1 8 8
echo -e "\n"
./sddmm_benchmark  /users/shigang/gitrepo/dlmc/rn50/random_pruning/0.9/bottleneck_2_block_group3_5_1.smtx 1024 2 1 1 1 8 8
echo -e "\n"
./sddmm_benchmark  /users/shigang/gitrepo/dlmc/rn50/random_pruning/0.9/bottleneck_2_block_group3_5_1.smtx 1024 4 1 1 1 8 8
echo -e "\n"
./sddmm_benchmark  /users/shigang/gitrepo/dlmc/rn50/random_pruning/0.9/bottleneck_2_block_group3_5_1.smtx 1024 8 1 1 1 8 8
echo -e "\n"
./sddmm_benchmark  /users/shigang/gitrepo/dlmc/rn50/random_pruning/0.95/bottleneck_2_block_group3_5_1.smtx 1024 2 1 1 1 8 8
echo -e "\n"
./sddmm_benchmark  /users/shigang/gitrepo/dlmc/rn50/random_pruning/0.95/bottleneck_2_block_group3_5_1.smtx 1024 4 1 1 1 8 8
echo -e "\n"
./sddmm_benchmark  /users/shigang/gitrepo/dlmc/rn50/random_pruning/0.95/bottleneck_2_block_group3_5_1.smtx 1024 8 1 1 1 8 8
echo -e "\n"
./sddmm_benchmark  /users/shigang/gitrepo/dlmc/rn50/random_pruning/0.98/bottleneck_2_block_group3_5_1.smtx 1024 2 1 1 1 8 8
echo -e "\n"
./sddmm_benchmark  /users/shigang/gitrepo/dlmc/rn50/random_pruning/0.98/bottleneck_2_block_group3_5_1.smtx 1024 4 1 1 1 8 8
echo -e "\n"
./sddmm_benchmark  /users/shigang/gitrepo/dlmc/rn50/random_pruning/0.98/bottleneck_2_block_group3_5_1.smtx 1024 8 1 1 1 8 8
echo -e "\n"


echo -e "L4-R4 \n"
./sddmm_benchmark  /users/shigang/gitrepo/dlmc/rn50/random_pruning/0.5/bottleneck_2_block_group3_5_1.smtx 1024 2 1 1 1 4 4
echo -e "\n"
./sddmm_benchmark  /users/shigang/gitrepo/dlmc/rn50/random_pruning/0.5/bottleneck_2_block_group3_5_1.smtx 1024 4 1 1 1 4 4
echo -e "\n"
./sddmm_benchmark  /users/shigang/gitrepo/dlmc/rn50/random_pruning/0.5/bottleneck_2_block_group3_5_1.smtx 1024 8 1 1 1 4 4
echo -e "\n"
./sddmm_benchmark  /users/shigang/gitrepo/dlmc/rn50/random_pruning/0.7/bottleneck_2_block_group3_5_1.smtx 1024 2 1 1 1 4 4
echo -e "\n"
./sddmm_benchmark  /users/shigang/gitrepo/dlmc/rn50/random_pruning/0.7/bottleneck_2_block_group3_5_1.smtx 1024 4 1 1 1 4 4
echo -e "\n"
./sddmm_benchmark  /users/shigang/gitrepo/dlmc/rn50/random_pruning/0.7/bottleneck_2_block_group3_5_1.smtx 1024 8 1 1 1 4 4
echo -e "\n"
./sddmm_benchmark  /users/shigang/gitrepo/dlmc/rn50/random_pruning/0.8/bottleneck_2_block_group3_5_1.smtx 1024 2 1 1 1 4 4
echo -e "\n"
./sddmm_benchmark  /users/shigang/gitrepo/dlmc/rn50/random_pruning/0.8/bottleneck_2_block_group3_5_1.smtx 1024 4 1 1 1 4 4
echo -e "\n"
./sddmm_benchmark  /users/shigang/gitrepo/dlmc/rn50/random_pruning/0.8/bottleneck_2_block_group3_5_1.smtx 1024 8 1 1 1 4 4
echo -e "\n"
./sddmm_benchmark  /users/shigang/gitrepo/dlmc/rn50/random_pruning/0.9/bottleneck_2_block_group3_5_1.smtx 1024 2 1 1 1 4 4
echo -e "\n"
./sddmm_benchmark  /users/shigang/gitrepo/dlmc/rn50/random_pruning/0.9/bottleneck_2_block_group3_5_1.smtx 1024 4 1 1 1 4 4
echo -e "\n"
./sddmm_benchmark  /users/shigang/gitrepo/dlmc/rn50/random_pruning/0.9/bottleneck_2_block_group3_5_1.smtx 1024 8 1 1 1 4 4
echo -e "\n"
./sddmm_benchmark  /users/shigang/gitrepo/dlmc/rn50/random_pruning/0.95/bottleneck_2_block_group3_5_1.smtx 1024 2 1 1 1 4 4
echo -e "\n"
./sddmm_benchmark  /users/shigang/gitrepo/dlmc/rn50/random_pruning/0.95/bottleneck_2_block_group3_5_1.smtx 1024 4 1 1 1 4 4
echo -e "\n"
./sddmm_benchmark  /users/shigang/gitrepo/dlmc/rn50/random_pruning/0.95/bottleneck_2_block_group3_5_1.smtx 1024 8 1 1 1 4 4
echo -e "\n"
./sddmm_benchmark  /users/shigang/gitrepo/dlmc/rn50/random_pruning/0.98/bottleneck_2_block_group3_5_1.smtx 1024 2 1 1 1 4 4
echo -e "\n"
./sddmm_benchmark  /users/shigang/gitrepo/dlmc/rn50/random_pruning/0.98/bottleneck_2_block_group3_5_1.smtx 1024 4 1 1 1 4 4
echo -e "\n"
./sddmm_benchmark  /users/shigang/gitrepo/dlmc/rn50/random_pruning/0.98/bottleneck_2_block_group3_5_1.smtx 1024 8 1 1 1 4 4
echo -e "\n"
