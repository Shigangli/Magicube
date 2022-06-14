## Magicube - SC22 Artifact

[Magicube](https://github.com/Shigangli/Magicube) is a high-performance library for quantized sparse matrix operations of deep learning on Tensor Cores. We conduct all the experiments on NVIDIA A100 GPU. The software requirements to reproduce the artfifact are: `CUDA Toolkit 11.4.0`, `Python 3.8.5`, `PyTorch 1.9.0` with `cuDNN version 8005`, which is a common configuration on a machine with A100 GPUs. All baselines, including vectorSparse (in SC21), are already integrated in the repo. The reproducibility process is easy to conduct following the steps below.

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

```bash
cd ./SDDMM/ablation_study
./compile_jobs.sh

# about 5 minutes
python sddmm_ablation_study.py > sddmm_abl_study.txt
``` 

**(4) To reproduce the results of Fig. 14:**

```bash
cd ./baselines
./setup.sh

# about 13 hours
./run_spmm_baselines.sh
``` 

```bash
cd ./SpMM/SpMM
./setup.sh

# about 8 hours
./run_spmm_magicube.sh
``` 

**(5) To reproduce the results of Fig. 15:**

```bash
cd ./baselines
./setup.sh

# about 8 hours
./run_sddmm_baselines.sh
``` 

```bash
cd ./SDDMM/SDDMM
./setup.sh

# about 5 hours
./run_sddmm_magicube.sh
``` 

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
```bash
cd ./plot

# generate csv files
./gen_csv.sh

# plot figures
./plot.sh
``` 
At last, see all figures in `./plot/figs`


<!--
## Alternate identifier:

https://github.com/Shigangli/Magicube
-->

Publication
-----------
Magicube will appear in [SC22](https://sc22.supercomputing.org/):
```bibtex
@inproceedings{li2022efficient,
  title={Efficient Quantized Sparse Matrix Operations on Tensor Cores},
  author={Li, Shigang and Osawa, Kazuki and Hoefler, Torsten},
  booktitle={International Conference for High Performance Computing, Networking, Storage and Analysis},
  year={2022}
}
```

## License

See [LICENSE](LICENSE).
