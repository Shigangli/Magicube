[![DOI](https://zenodo.org/badge/351565020.svg)](https://zenodo.org/badge/latestdoi/351565020)
#### Grant access to GPU performance counter
We obtain the kernel duration with [NVIDIA Nsight Compute](https://developer.nvidia.com/nsight-compute). Profiling with Nsight Compute requires access to the performance counters on the GPU ([Permission issue with Performance Counters](https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters)). This should be configured on the machine outside of the container.

To check if the performance is accessible, run
```shell
cat /proc/driver/nvidia/params | grep RmProfilingAdminOnly
```
You should see
```shell
RmProfilingAdminOnly: 0
```
Otherwise, the access can be granted with the following steps:
1. Create .conf file (e.g. profile.conf) in folder /etc/modprobe.d
2. Open file /etc/modprobe.d/profile.conf in any editor
3. Add below line in profile.conf
   ```
   options nvidia “NVreg_RestrictProfilingToAdminUsers=0”
   ```
4. Close file /etc/modprobe.d/profile.conf
5. Restart your machine

For more information, please see [nvprof-warning-the-user-does-not-have-permission-to-profile-on-the-target-device](https://forums.developer.nvidia.com/t/nvprof-warning-the-user-does-not-have-permission-to-profile-on-the-target-device/72374/8) and [Permission issue with Performance Counters](https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters).

#### Using Docker

Step 1: Get the source code
```shell
git clone https://github.com/apuaaChen/vectorSparse.git && cd vectorSparse
```

Step 2: We provides a Dockerfile that builds the proper environment with all dependencies. Note that nvidia-docker must be installed to run on GPU. To build the image, run the following command:
```shell
docker build -t vectorsparse .
```

Step 3: Get The dataset
We use the [Deep Learning Matrix Collection](https://storage.googleapis.com/sgk-sc2020/dlmc.tar.gz). Please download the dataset and put it into the directory <host_dataset_dir>. The directory will be something like  <host_dataset_dir>/dlmc/rn50/....

Step 4: To launch the container
```shell
docker run -it --gpus all --name <your_container_name> -v <host_dataset_dir>:/raid/datasets -v <host_dir>/vectorSparse:/projects/vectorSparse vectorsparse
```
So that in the container, the sparse matrices will be available at /raid/datasets/dlmc/rn50/...

Step 5: Compile the source code with
```shell
cd vectorSparse
bash setup.sh
```

Step 6.1: To obtain the results in Figure 17, run
```shell
python3 launch.py --exp spmm
```
This script will launch all the experiments sequentially. For each experiment, the profiling result is stored in a `.csv` file under the `./csv` directory. We present an example csv file in the `./example`. When all the experiments are done, another python script will be lauched to fetch the kernel durations in the csv files, and summarize the results as a figure in `spmm_speedup_rn50_combo.pdf`.

Step 6.2: To obtain the results in Figure 18, run
```shell
python3 launch.py --exp sddmm
```
The result will be shown in sddmm_speedup_rn50_combo.pdf

***

The DLMC dataset and sputnik library are from this paper
```
@inproceedings{sgk_sc2020,
  author    = {Trevor Gale and Matei Zaharia and Cliff Young and Erich Elsen},
  title     = {Sparse {GPU} Kernels for Deep Learning},
  booktitle = {Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis, {SC} 2020},
  year      = {2020},
}
```
***
We demonstrate how to use our kernels in Sparse Transformer with fixed mask in [here](https://github.com/apuaaChen/sparse_transformer_sc21).
