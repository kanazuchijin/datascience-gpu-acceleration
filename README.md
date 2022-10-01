# datascience-gpu-acceleration
This is an experiment to assess the performance benefits of GPU acceleration using CUDA-based NVIDIA RTX video cards. Many data science researchers use their personal computers to perform modeling and analysis. While supercomputers are preferred for larger data applications, access to a supercomputer lies out of reach for the majority of researchers. As an alternative, researchers can utilize GPU accelerated architecture to reduce processing time. Yet, what is not understood is how much benefit can be derived from GPU acceleration.

In considering the type of graphics cards used in this study, since the target audience includes desktops and laptops found in most offices, we consider GPUs marketed toward the consumer space. Another consideration is the data modeling platform. We have chosen R and Python. 

In this experiment, four conditions are tested:
1. Non-GPU accelerated platform: This condition utilizes hardware without a graphics card. This serves as a baseline.
1. NVIDIA RTX 3060Ti: This is a lower end GPU with CUDA cores and represents a "typical" graphics card found in the majority of desktops and laptops.
1. NVIDIA RTX 4090: The most powerful graphics card on the market with CUDA cores aimed at the consumer market.
1. Supercomputer: 



Benefits for Python because no need for `cython` [cython](https://cython.org/).

[CUDA parallel computing GPU libraries](https://developer.nvidia.com/gpu-accelerated-libraries)