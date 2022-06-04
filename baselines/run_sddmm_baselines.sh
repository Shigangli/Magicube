
echo "Tesing sddmm_cublas_fp16"
python launch_sddmm_cublas_fp16.py > sddmm_cublas_fp16.txt
echo "Finish sddmm_cublas_fp16"

echo "Tesing sddmm_cublas_int8"
python launch_sddmm_cublas_int8.py > sddmm_cublas_int8.txt
echo "Finish sddmm_cublas_int8"

echo "Tesing sddmm_vectorSparse"
python launch_sddmm_vectorSparse.py > sddmm_vectorSparse.txt
echo "Finish sddmm_vectorSparse"
