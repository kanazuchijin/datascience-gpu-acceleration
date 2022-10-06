#include <iostream>
#include <math.h>

// function to add the elements of two arrays
/* I have to turn our add function into a function that the 
GPU can run, called a kernel in CUDA. To do this, all I have 
to do is add the specifier __global__ to the function, which 
tells the CUDA C++ compiler that this is a function that runs 
on the GPU and can be called from CPU code. 

These __global__ functions are known as kernels, and code that 
runs on the GPU is often called device code, while code that 
runs on the CPU is host code.*/
__global__
void add(int n, float *x, float *y)
{
  for (int i = 0; i < n; i++)
      y[i] = x[i] + y[i];
}

int main(void)
{
  int N = 1<<20; // 1M elements

  // CPU code; this is replaced by GPU code below
  //float *x = new float[N];
  //float *y = new float[N];

  /* The GPU equivalent of the code above to allocate 
  memory accessible by the GPU. The call to the
  function cudaMallocManaged() allocates data in
  unified memory. The function returns a pointer
  that is accessible by host (CPU) or device (GPU) code. */
  float *x, *y;
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // Run kernel on 1M elements on the CPU
  //add(N, x, y);

  // Run kernel on 1M elements on the GPU
  add<<<1, 1>>>(N, x, y);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  std::cout << "Max error: " << maxError << std::endl;

  // Free memory
  // CPU specific code; Replaced by code below
  //delete [] x;
  //delete [] y;

  // Free memory
  // GPU specific code to pass the pointer
  cudaFree(x);
  cudaFree(x);

  return 0;
}