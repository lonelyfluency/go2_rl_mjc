# Go2 Mujoco RL

## setting up environment

Anaconda is recommend for installing the environment.

1. Create conda env:
    ```bash
    conda create -n mjx python=3.10
    conda activate mjx
    ```
2. Install python libs:
    ```bash
    pip install scipy
    pip install matplotlib
    pip install opencv-python
    pip install mujoco
    conda install jaxlib=*=*cuda* jax cuda-nvcc -c conda-forge -c nvidia
    pip install brax
    pip install ipython
    pip install tqdm
    ```
3. Test:
    ```bash
    python test.py
    ```
    If it successfully output a video output.mp4, you are good.

4. Depending on the CUDA version of your machine( wheels only available on linux ), run EITHER of the following:

 CUDA 12.X installation
 ```bash
 pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
 ```
OR

 CUDA 11.X installation
 Note: wheels only available on linux.
 ```bash
 pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
 ```
 To double check if you have have successfully configured the gpu:
 ```bash
 python -c "import jax; print(f'Jax backend: {jax.default_backend()}')"
 Jax backend: gpu 
 ```
## normal rl training

run train_go2.py for normal rl training.
