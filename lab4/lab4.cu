#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <string>
#include <unistd.h>
#include <ctime>
#include <thrust/extrema.h>
#include <thrust/device_vector.h>
#include <iomanip>

using namespace std;

#define CSC(call) \
do { \
    cudaError_t status = call; \
    if (status != cudaSuccess) { \
        fprintf(stderr, "ERROR is %s:%d. Message: %s\n", __FILE__, __LINE__, cudaGetErrorString(status)); \
        exit(0); \
    } \
} while(0)



struct comparator {                                             
    __device__ bool operator()(double a, double b) {     
        return abs(a) < abs(b);                              
    }
};

__global__ void swap_str(double *data, int idxm, int i, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = blockDim.x * gridDim.x;

    while (idx < n) {

        double sw = data[idx * n + i];
        data[idx * n + i] = data[idx * n + idxm];
        data[idx * n + idxm] = sw;


        idx += offset;
    }
}


__global__ void kernel_1(double *data, int n, int j) { 

    int idx = blockIdx.x * blockDim.x + threadIdx.x + j + 1;
    int offset = blockDim.x * gridDim.x;

    while (idx < n) {
        
        data[n * j + idx] = data[n * j + idx] / data[n * j + j];

        idx += offset;
    }
}


__global__ void kernel_2(double *data, int n, int jj) {

    int idy = blockDim.x * blockIdx.x + threadIdx.x + jj + 1;
    int idx = blockDim.y * blockIdx.y + threadIdx.y  + jj + 1;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;


    for(int i = idy; i < n; i += offsety) {
        for(int j = idx; j < n; j += offsetx) {
            data[j * n + i] = data[j * n + i] - data[j * n + jj] * data[jj * n + i];         
        }
    }
}


int main() {
    int n;

    ios::sync_with_stdio(false);
    cin.tie(0);

    cin >> n;
    double *data = (double *)malloc(sizeof(double) * n * n);
    int *p = (int *)malloc(sizeof(int) * n);


    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            cin >> data[j * n + i];
        }
    }

    double *dev_data;

    CSC(cudaMalloc(&dev_data, sizeof(double) * n * n));
    CSC(cudaMemcpy(dev_data, data, sizeof(double) * n * n, cudaMemcpyHostToDevice));

    comparator comp;

    for (int j = 0; j < n; ++j) {

        thrust::device_ptr<double> p_data = thrust::device_pointer_cast(dev_data);
        thrust::device_ptr<double> res = thrust::max_element(p_data + n * j + j, p_data + n * (j + 1), comp);
        p[j] = (int) (res - p_data) - n * j;
        swap_str<<<1024, 1024>>>(dev_data, p[j], j, n);
        kernel_1<<<1024, 1024>>> (dev_data, n, j);
        kernel_2<<<dim3(32, 32), dim3(32, 32)>>> (dev_data, n, j);
    }

    cout << setprecision(10) << fixed;
    CSC(cudaMemcpy(data, dev_data, sizeof(double) * n * n, cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            cout << data[j * n + i] << " ";
        }
        cout << endl;
    }

    cout << resetiosflags(ios::fixed);
    for (int i = 0; i < n; ++i) {
        cout << p[i] << " ";
    }
    cout << endl;

    CSC(cudaFree(dev_data));
    free(data);
    free(p);
	return 0;
}
