
echo "Tesing spmm_cublas_fp16"
python launch_spmm_cublas_fp16.py > spmm_cublas_fp16.txt
echo "Finish spmm_cublas_fp16"

echo "Tesing spmm_cublas_int8"
python launch_spmm_cublas_int8.py > spmm_cublas_int8.txt
echo "Finish spmm_cublas_int8"

echo "Tesing spmm_vectorSparse"
python launch_spmm_vectorSparse.py > spmm_vectorSparse.txt
echo "Finish spmm_vectorSparse"

echo "Tesing spmm_cusparse_fp16"
python launch_spmm_cusparse_fp16.py > spmm_cusparse_fp16.txt
echo "Finish spmm_cusparse_fp16"

echo "Tesing spmm_cusparse_int8"
python launch_spmm_cusparse_int8.py > spmm_cusparse_int8.txt
echo "Finish spmm_cusparse_int8"
