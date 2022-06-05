
echo -e "Evaluation perf for different precisions: N = 512, Iteration = 1024 \n"
echo -e "L16-R16 \n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.5/bottleneck_2_block_group3_5_1.smtx 512 2 0 1 1 1 16 16
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.5/bottleneck_2_block_group3_5_1.smtx 512 4 0 1 1 1 16 16
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.5/bottleneck_2_block_group3_5_1.smtx 512 8 0 1 1 1 16 16
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.7/bottleneck_2_block_group3_5_1.smtx 512 2 0 1 1 1 16 16
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.7/bottleneck_2_block_group3_5_1.smtx 512 4 0 1 1 1 16 16
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.7/bottleneck_2_block_group3_5_1.smtx 512 8 0 1 1 1 16 16
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.8/bottleneck_2_block_group3_5_1.smtx 512 2 0 1 1 1 16 16
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.8/bottleneck_2_block_group3_5_1.smtx 512 4 0 1 1 1 16 16
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.8/bottleneck_2_block_group3_5_1.smtx 512 8 0 1 1 1 16 16
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.9/bottleneck_2_block_group3_5_1.smtx 512 2 0 1 1 1 16 16
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.9/bottleneck_2_block_group3_5_1.smtx 512 4 0 1 1 1 16 16
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.9/bottleneck_2_block_group3_5_1.smtx 512 8 0 1 1 1 16 16
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.95/bottleneck_2_block_group3_5_1.smtx 512 2 0 1 1 1 16 16
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.95/bottleneck_2_block_group3_5_1.smtx 512 4 0 1 1 1 16 16
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.95/bottleneck_2_block_group3_5_1.smtx 512 8 0 1 1 1 16 16
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.98/bottleneck_2_block_group3_5_1.smtx 512 2 0 1 1 1 16 16
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.98/bottleneck_2_block_group3_5_1.smtx 512 4 0 1 1 1 16 16
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.98/bottleneck_2_block_group3_5_1.smtx 512 8 0 1 1 1 16 16
echo -e "\n"

echo -e "L16-R8 \n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.5/bottleneck_2_block_group3_5_1.smtx 512 2 0 1 1 1 16 8
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.5/bottleneck_2_block_group3_5_1.smtx 512 4 0 1 1 1 16 8
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.5/bottleneck_2_block_group3_5_1.smtx 512 8 0 1 1 1 16 8
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.7/bottleneck_2_block_group3_5_1.smtx 512 2 0 1 1 1 16 8
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.7/bottleneck_2_block_group3_5_1.smtx 512 4 0 1 1 1 16 8
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.7/bottleneck_2_block_group3_5_1.smtx 512 8 0 1 1 1 16 8
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.8/bottleneck_2_block_group3_5_1.smtx 512 2 0 1 1 1 16 8
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.8/bottleneck_2_block_group3_5_1.smtx 512 4 0 1 1 1 16 8
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.8/bottleneck_2_block_group3_5_1.smtx 512 8 0 1 1 1 16 8
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.9/bottleneck_2_block_group3_5_1.smtx 512 2 0 1 1 1 16 8
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.9/bottleneck_2_block_group3_5_1.smtx 512 4 0 1 1 1 16 8
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.9/bottleneck_2_block_group3_5_1.smtx 512 8 0 1 1 1 16 8
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.95/bottleneck_2_block_group3_5_1.smtx 512 2 0 1 1 1 16 8
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.95/bottleneck_2_block_group3_5_1.smtx 512 4 0 1 1 1 16 8
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.95/bottleneck_2_block_group3_5_1.smtx 512 8 0 1 1 1 16 8
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.98/bottleneck_2_block_group3_5_1.smtx 512 2 0 1 1 1 16 8
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.98/bottleneck_2_block_group3_5_1.smtx 512 4 0 1 1 1 16 8
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.98/bottleneck_2_block_group3_5_1.smtx 512 8 0 1 1 1 16 8
echo -e "\n"

echo -e "L8-R8 \n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.5/bottleneck_2_block_group3_5_1.smtx 512 2 0 1 1 1 8 8
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.5/bottleneck_2_block_group3_5_1.smtx 512 4 0 1 1 1 8 8
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.5/bottleneck_2_block_group3_5_1.smtx 512 8 0 1 1 1 8 8
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.7/bottleneck_2_block_group3_5_1.smtx 512 2 0 1 1 1 8 8
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.7/bottleneck_2_block_group3_5_1.smtx 512 4 0 1 1 1 8 8
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.7/bottleneck_2_block_group3_5_1.smtx 512 8 0 1 1 1 8 8
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.8/bottleneck_2_block_group3_5_1.smtx 512 2 0 1 1 1 8 8
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.8/bottleneck_2_block_group3_5_1.smtx 512 4 0 1 1 1 8 8
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.8/bottleneck_2_block_group3_5_1.smtx 512 8 0 1 1 1 8 8
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.9/bottleneck_2_block_group3_5_1.smtx 512 2 0 1 1 1 8 8
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.9/bottleneck_2_block_group3_5_1.smtx 512 4 0 1 1 1 8 8
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.9/bottleneck_2_block_group3_5_1.smtx 512 8 0 1 1 1 8 8
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.95/bottleneck_2_block_group3_5_1.smtx 512 2 0 1 1 1 8 8
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.95/bottleneck_2_block_group3_5_1.smtx 512 4 0 1 1 1 8 8
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.95/bottleneck_2_block_group3_5_1.smtx 512 8 0 1 1 1 8 8
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.98/bottleneck_2_block_group3_5_1.smtx 512 2 0 1 1 1 8 8
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.98/bottleneck_2_block_group3_5_1.smtx 512 4 0 1 1 1 8 8
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.98/bottleneck_2_block_group3_5_1.smtx 512 8 0 1 1 1 8 8
echo -e "\n"

echo -e "L16-R4 \n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.5/bottleneck_2_block_group3_5_1.smtx 512 2 0 1 1 1 16 4
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.5/bottleneck_2_block_group3_5_1.smtx 512 4 0 1 1 1 16 4
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.5/bottleneck_2_block_group3_5_1.smtx 512 8 0 1 1 1 16 4
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.7/bottleneck_2_block_group3_5_1.smtx 512 2 0 1 1 1 16 4
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.7/bottleneck_2_block_group3_5_1.smtx 512 4 0 1 1 1 16 4
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.7/bottleneck_2_block_group3_5_1.smtx 512 8 0 1 1 1 16 4
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.8/bottleneck_2_block_group3_5_1.smtx 512 2 0 1 1 1 16 4
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.8/bottleneck_2_block_group3_5_1.smtx 512 4 0 1 1 1 16 4
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.8/bottleneck_2_block_group3_5_1.smtx 512 8 0 1 1 1 16 4
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.9/bottleneck_2_block_group3_5_1.smtx 512 2 0 1 1 1 16 4
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.9/bottleneck_2_block_group3_5_1.smtx 512 4 0 1 1 1 16 4
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.9/bottleneck_2_block_group3_5_1.smtx 512 8 0 1 1 1 16 4
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.95/bottleneck_2_block_group3_5_1.smtx 512 2 0 1 1 1 16 4
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.95/bottleneck_2_block_group3_5_1.smtx 512 4 0 1 1 1 16 4
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.95/bottleneck_2_block_group3_5_1.smtx 512 8 0 1 1 1 16 4
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.98/bottleneck_2_block_group3_5_1.smtx 512 2 0 1 1 1 16 4
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.98/bottleneck_2_block_group3_5_1.smtx 512 4 0 1 1 1 16 4
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.98/bottleneck_2_block_group3_5_1.smtx 512 8 0 1 1 1 16 4
echo -e "\n"

echo -e "L12-R4 \n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.5/bottleneck_2_block_group3_5_1.smtx 512 2 0 1 1 1 12 4
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.5/bottleneck_2_block_group3_5_1.smtx 512 4 0 1 1 1 12 4
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.5/bottleneck_2_block_group3_5_1.smtx 512 8 0 1 1 1 12 4
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.7/bottleneck_2_block_group3_5_1.smtx 512 2 0 1 1 1 12 4
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.7/bottleneck_2_block_group3_5_1.smtx 512 4 0 1 1 1 12 4
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.7/bottleneck_2_block_group3_5_1.smtx 512 8 0 1 1 1 12 4
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.8/bottleneck_2_block_group3_5_1.smtx 512 2 0 1 1 1 12 4
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.8/bottleneck_2_block_group3_5_1.smtx 512 4 0 1 1 1 12 4
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.8/bottleneck_2_block_group3_5_1.smtx 512 8 0 1 1 1 12 4
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.9/bottleneck_2_block_group3_5_1.smtx 512 2 0 1 1 1 12 4
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.9/bottleneck_2_block_group3_5_1.smtx 512 4 0 1 1 1 12 4
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.9/bottleneck_2_block_group3_5_1.smtx 512 8 0 1 1 1 12 4
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.95/bottleneck_2_block_group3_5_1.smtx 512 2 0 1 1 1 12 4
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.95/bottleneck_2_block_group3_5_1.smtx 512 4 0 1 1 1 12 4
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.95/bottleneck_2_block_group3_5_1.smtx 512 8 0 1 1 1 12 4
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.98/bottleneck_2_block_group3_5_1.smtx 512 2 0 1 1 1 12 4
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.98/bottleneck_2_block_group3_5_1.smtx 512 4 0 1 1 1 12 4
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.98/bottleneck_2_block_group3_5_1.smtx 512 8 0 1 1 1 12 4
echo -e "\n"

echo -e "L8-R4 \n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.5/bottleneck_2_block_group3_5_1.smtx 512 2 0 1 1 1 8 4
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.5/bottleneck_2_block_group3_5_1.smtx 512 4 0 1 1 1 8 4
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.5/bottleneck_2_block_group3_5_1.smtx 512 8 0 1 1 1 8 4
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.7/bottleneck_2_block_group3_5_1.smtx 512 2 0 1 1 1 8 4
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.7/bottleneck_2_block_group3_5_1.smtx 512 4 0 1 1 1 8 4
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.7/bottleneck_2_block_group3_5_1.smtx 512 8 0 1 1 1 8 4
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.8/bottleneck_2_block_group3_5_1.smtx 512 2 0 1 1 1 8 4
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.8/bottleneck_2_block_group3_5_1.smtx 512 4 0 1 1 1 8 4
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.8/bottleneck_2_block_group3_5_1.smtx 512 8 0 1 1 1 8 4
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.9/bottleneck_2_block_group3_5_1.smtx 512 2 0 1 1 1 8 4
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.9/bottleneck_2_block_group3_5_1.smtx 512 4 0 1 1 1 8 4
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.9/bottleneck_2_block_group3_5_1.smtx 512 8 0 1 1 1 8 4
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.95/bottleneck_2_block_group3_5_1.smtx 512 2 0 1 1 1 8 4
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.95/bottleneck_2_block_group3_5_1.smtx 512 4 0 1 1 1 8 4
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.95/bottleneck_2_block_group3_5_1.smtx 512 8 0 1 1 1 8 4
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.98/bottleneck_2_block_group3_5_1.smtx 512 2 0 1 1 1 8 4
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.98/bottleneck_2_block_group3_5_1.smtx 512 4 0 1 1 1 8 4
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.98/bottleneck_2_block_group3_5_1.smtx 512 8 0 1 1 1 8 4
echo -e "\n"

echo -e "L4-R4 \n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.5/bottleneck_2_block_group3_5_1.smtx 512 2 0 1 1 1 4 4
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.5/bottleneck_2_block_group3_5_1.smtx 512 4 0 1 1 1 4 4
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.5/bottleneck_2_block_group3_5_1.smtx 512 8 0 1 1 1 4 4
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.7/bottleneck_2_block_group3_5_1.smtx 512 2 0 1 1 1 4 4
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.7/bottleneck_2_block_group3_5_1.smtx 512 4 0 1 1 1 4 4
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.7/bottleneck_2_block_group3_5_1.smtx 512 8 0 1 1 1 4 4
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.8/bottleneck_2_block_group3_5_1.smtx 512 2 0 1 1 1 4 4
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.8/bottleneck_2_block_group3_5_1.smtx 512 4 0 1 1 1 4 4
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.8/bottleneck_2_block_group3_5_1.smtx 512 8 0 1 1 1 4 4
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.9/bottleneck_2_block_group3_5_1.smtx 512 2 0 1 1 1 4 4
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.9/bottleneck_2_block_group3_5_1.smtx 512 4 0 1 1 1 4 4
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.9/bottleneck_2_block_group3_5_1.smtx 512 8 0 1 1 1 4 4
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.95/bottleneck_2_block_group3_5_1.smtx 512 2 0 1 1 1 4 4
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.95/bottleneck_2_block_group3_5_1.smtx 512 4 0 1 1 1 4 4
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.95/bottleneck_2_block_group3_5_1.smtx 512 8 0 1 1 1 4 4
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.98/bottleneck_2_block_group3_5_1.smtx 512 2 0 1 1 1 4 4
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.98/bottleneck_2_block_group3_5_1.smtx 512 4 0 1 1 1 4 4
echo -e "\n"
./spmm_benchmark  ${dataset_dir}/rn50/random_pruning/0.98/bottleneck_2_block_group3_5_1.smtx 512 8 0 1 1 1 4 4
echo -e "\n"
