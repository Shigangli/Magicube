nvcc u4_mma.cu -ccbin g++ -m64 -gencode arch=compute_80,code=compute_80 -I./Common -o test1 -Dverify_output
