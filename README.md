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
    pip install tqdm
    ```
3. Test:
    ```bash
    python test.py
    ```
    If it successfully output a video output.mp4, you are good.

## normal rl training

run train_go2.py for normal rl training.