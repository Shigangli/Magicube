## Magicube

The software requirements to reproduce the artfifact are: `CUDA Toolkit 11.4.0`, `Python 3.8.5`, `PyTorch 1.9.0` with `cuDNN version 8005`,

which is a common configuration on a machine with A100 GPUs. This is no complex software dependency and the reproducibility process is easy to conduct following the steps below.

## Step 1: Prepare Dataset and Code.

```bash
wget https://storage.googleapis.com/sgk-sc2020/dlmc.tar.gz
tar -xvf dlmc.tar.gz
export dataset_dir=/the/path/of/dlmc
git clone git@github.com:Shigangli/Magicube.git
``` 

## Step 2: Setup Python environment.

```bash
conda create --name py38_sc22 python=3.8
conda activate py38_sc22
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
``` 

## Step 3: Compile and run the experiments.

**(1) To reproduce the results of Fig. 11:**

```bash
cd ./SpMM/ablation_study

# about 3 minutes
./compile_jobs.sh

# about 3 minutes
./spmm_ablation_study.sh > spmm_abl_study.txt
``` 

**(2) To reproduce the results of Fig. 12:**

```bash
cd ./SpMM/SpMM
./setup.sh

# about 5 minutes
./spmm_pres.sh > spmm_pres.txt
``` 

**(3) To reproduce the results of Fig. 13:**

`cd ./SDDMM/ablation_study`

`./compile_jobs.sh`

`python sddmm_ablation_study.py`

**(4) To reproduce the results of Fig. 14:**

`cd ./baselines`

`./setup.sh`

`python launch_spmm_cublas_fp16.py`

`python launch_spmm_cublas_int8.py`

`python launch_spmm_vectorSparse.py`

`python launch_spmm_cusparse_fp16.py`

`python launch_spmm_cusparse_int8.py`

`cd ./SpMM/SpMM`

`./setup.sh`

`python launch_spmm_magicube_16b8b.py`

`python launch_spmm_magicube_8b8b.py`

`python launch_spmm_magicube_8b4b.py`

`python launch_spmm_magicube_4b4b.py`

**(5) To reproduce the results of Fig. 15:**

`cd ./baselines`

`./setup.sh`

`python launch_sddmm_cublas_fp16.py`

`python launch_sddmm_cublas_int8.py`

`python launch_sddmm_vectorSparse.py`

cd ./SDDMM/SDDMM

`./setup.sh`

`python launch_sddmm_magicube_16b16b.py`

`python launch_sddmm_magicube_8b8b.py`

`python launch_sddmm_magicube_4b4b.py`

**(6) To reproduce the results of Fig. 16:**
  
```bash
cd ./end2end_eval/sparse_transformer_baselines/src/
./install.sh
cd ..

# Run Pytorch (cudnn) end-to-end test, which takes about 0.5 hour. 
# It's all right if it reports Out-of-Memory error during the evaluation.
python launch_cudnn_fp16.py > pytorch_n2n.txt

# Run vectorSparse end-to-end test, which takes about 0.8 hour.
python launch_vectorSparse.py > vectorSparse_n2n.txt
``` 

```bash
cd ./end2end_eval/sparse_transformer_magicube/src
./install.sh
cd ..

# Run magicube end-to-end test, which takes about 2.6 hours.
python launch_magicube.py > magicube_n2n.txt
``` 


## Step 4: Plot the figures.
Figure 13
Figure 14
Figure 15
Figure 16


## License

See [LICENSE](LICENSE).
