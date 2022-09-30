#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>

using namespace std;

#define CSC(call) \
do { \
    cudaError_t status = call; \
    if (status != cudaSuccess) { \
        fprintf(stderr, "ERROR is %s:%d. Message: %s\n", __FILE__, __LINE__, cudaGetErrorString(status)); \
        exit(0); \
    } \
} while(0)

__global__ void kernel(double *v12, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = blockDim.x * gridDim.x;
    while (idx < n) {
        v12[idx] = max(v12[idx], v12[idx + n]);
        idx += offset;
    }
}

int main() {

    long long n;
    cin >> n;

    double *v12 = (double *)malloc(sizeof(double) * n * 2);

    for (int i = 0; i < 2 * n; ++i) {
        cin >> v12[i];
    }

    double *dev_arr;
    CSC(cudaMalloc(&dev_arr, sizeof(double) * n * 2));
    CSC(cudaMemcpy(dev_arr, v12, sizeof(double) * n * 2, cudaMemcpyHostToDevice));

    kernel<<<1024, 1024>>>(dev_arr, n);

    CSC(cudaDeviceSynchronize());
    CSC(cudaGetLastError());

    CSC(cudaMemcpy(v12, dev_arr, sizeof(double) * n * 2, cudaMemcpyDeviceToHost));
    for(int i = 0; i < n; ++i)
        cout << v12[i] << " ";
    cout << endl;

    CSC(cudaFree(dev_arr));
    free(v12);
    return 0;
}