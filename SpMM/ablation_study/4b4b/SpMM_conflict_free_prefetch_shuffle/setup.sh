rm ./bin/*
#rm sddmm_benchmark
rm spmm_benchmark

mkdir -p ./bin
#make sddmm_benchmark
make spmm_benchmark
