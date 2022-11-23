## Magicube: Efficient Quantized Sparse Matrix Operations on Tensor Cores

![Magicube Logo](magicubeLogo.svg)

Magicube is a high-performance library for quantized sparse matrix operations (SpMM and SDDMM) of deep learning on Tensor Cores. Magicube is published in [SC 2022](https://sc22.supercomputing.org/), Best Paper Finalist. We conduct all the experiments on NVIDIA A100-SXM4-40GB GPU. The software requirements to reproduce the artfifact are: `GCC 8.4.1`, `CUDA Toolkit 11.4.0`, `Python 3.8.5`, `PyTorch 1.9.0` with `cuDNN version 8005`.

**We provide two ways to reproduce the results.** The first way is to reproduce the artifact with **docker container**, in which the software environment is already configured and the input dataset is also included. Note that nvidia-docker must be installed to run the container on GPU. Using docker container enables an easy reproducibility process. The second way is to reproduce the artifact with **source code**, in which users have to setup the software environment and download the input dataset by themselves following the provided instructions.

## Reproduction with container

We run all the experiments on NVIDIA A100-SXM4-40GB GPU. Please double-check the model of GPU by `nvidia-smi -L`. Note that nvidia-docker must be installed to run the container on GPU. Use the following three steps to reproduce the artifact with docker container.

**Step 1:** Download and run the container.

Download magicube_container.tar.gz from the DOI by:

```bash
wget https://zenodo.org/record/6924338/files/magicube_container.tar.gz
```

Run the container and activate python by:

```bash
docker load -i magicube_container.tar.gz
docker run -it --gpus all magicube_container
source /artifacts/sc22_venv/bin/activate
```

**Step 2:** Compile and run the experiments.

**(1)** To reproduce the results of Fig. 11:

```bash
cd /artifacts/Magicube/SpMM/ablation_study

# about 3 minutes
bash compile_jobs.sh

# about 3 minutes
bash spmm_ablation_study.sh > spmm_abl_study.txt
```

**(2)** To reproduce the results of Fig. 12:

```bash
cd /artifacts/Magicube/SpMM/SpMM

bash setup.sh

# about 5 minutes
bash spmm_pres.sh > spmm_pres.txt
```

**(3)** To reproduce the results of Fig. 13:

```bash
cd /artifacts/Magicube/SDDMM/ablation_study

bash compile_jobs.sh

# about 5 minutes
python sddmm_ablation_study.py > sddmm_abl_study.txt
```

**(4)** To reproduce the results of Fig. 14:

```bash
cd /artifacts/Magicube/baselines
bash setup.sh

# about 13 hours
bash run_spmm_baselines.sh

cd /artifacts/Magicube/SpMM/SpMM
bash setup.sh

# about 8 hours
bash run_spmm_magicube.sh
```

**(5)** To reproduce the results of Fig. 15:

```bash
cd /artifacts/Magicube/baselines
bash setup.sh

# about 8 hours
bash run_sddmm_baselines.sh

cd /artifacts/Magicube/SDDMM/SDDMM
bash setup.sh

# about 5 hours
bash run_sddmm_magicube.sh
```

**(6)** To reproduce the results of Fig. 16:

```bash
cd /artifacts/Magicube/end2end_eval/ sparse_transformer_baselines/src
bash install.sh
cd ..

# about 0.5 hour
python launch_cudnn_fp16.py > pytorch_n2n.txt

# about 0.8 hour
python launch_vectorSparse.py > vectorSparse_n2n.txt

cd /artifacts/Magicube/end2end_eval/sparse_transformer_magicube/src
bash install.sh
cd ..

# about 2.6 hours
python launch_magicube.py > magicube_n2n.txt
```

**Step 3**: Plot the figures.

```bash
cd /artifacts/Magicube/plot

# generate csv files
bash gen_csv.sh

# plot figures
bash plot.sh

# copy figures
cd /artifacts/Magicube/plot/figs
scp *.pdf username@hostmachine:/host/path/target
```

## Reproduction with source code

Different from docker container, users have to setup the software environment and download the input dataset by themselves when reproducing from source code.

**Step 1**: Prepare dataset and code, and setup python environment.

Download input dataset and source code:

```bash
wget https://storage.googleapis.com/sgk-sc2020/dlmc.tar.gz
tar -xvf dlmc.tar.gz
export dataset_dir=/the/path/of/dlmc 
git clone git@github.com:Shigangli/Magicube.git
```

Setup python environment:

```bash
conda create --name py38_sc22 python=3.8
conda activate py38_sc22
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

**Steps 2&3**: Suppose the source code is in the path of `/artifacts/Magicube/`. Then, follow the same Steps 2&3 as reproduction with container to reproduce the results and figures.

## Publication

Magicube is pulished in SC 2022, Best Paper Finalist. To cite our work:
```bibtex
@inproceedings{li2022efficient,
  author = {Li, Shigang and Osawa, Kazuki and Hoefler, Torsten},
  title = {Efficient Quantized Sparse Matrix Operations on Tensor Cores},
  booktitle = {Proceedings of the International Conference on High Performance Computing, Networking, Storage and Analysis},
  articleno = {37},
  numpages = {15},
  location = {Dallas, Texas},
  publisher = {IEEE Press},
  series = {SC'22},
  year = {2022}
}
```

## License

See [LICENSE](LICENSE).
