#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <string>

using namespace std;

#define CSC(call) \
do { \
    cudaError_t status = call; \
    if (status != cudaSuccess) { \
        fprintf(stderr, "ERROR is %s:%d. Message: %s\n", __FILE__, __LINE__, cudaGetErrorString(status)); \
        exit(0); \
    } \
} while(0)


__device__ double conv(double r, double g, double b) {
    return 0.299 * r + 0.587 * g + 0.114 * b;
}

__global__ void kernel(cudaTextureObject_t texObj, uchar4 *out, int w, int h) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;
    int x, y;
    uchar4 p;

    double mx[3][3] = {
        {-1, 0, 1},
        {-1, 0, 1},
        {-1, 0, 1}
    };

    double my[3][3] = {
        {-1, -1, -1},
        {0, 0, 0},
        {1, 1, 1}
    };

    for(y = idy; y < h; y += offsety) {
        for(x = idx; x < w; x += offsetx) {
            
            double gx = 0;
            double gy = 0;

            for (int i = -1; i < 2; ++i) {
                for (int j = -1; j < 2; ++j) {

                    p = tex2D<uchar4>(texObj, x + i, y + j);

                    gx += mx[i + 1][j + 1] * conv(p.x, p.y, p.z);
                    gy += my[i + 1][j + 1] * conv(p.x, p.y, p.z);

                }
            }

            int g = min(255, int(sqrt(gx * gx + gy * gy)));

            out[y * w + x] = make_uchar4(g, g, g, p.w);
        }
    }
}

int main() {

    string in_name, out_name;
    cin >> in_name >> out_name;
    int w, h;

    FILE *in;

    if ((in = fopen(in_name.c_str(), "rb")) == NULL) {
        cout << "File open error\n";
        return -1;
    }

    fread(&w, sizeof(int), 1, in);
    fread(&h, sizeof(int), 1, in);
    uchar4 *data = (uchar4 *)malloc(sizeof(uchar4) * w * h);
    fread(data, sizeof(uchar4), w * h, in);


    cudaArray *arr;
    cudaChannelFormatDesc ch = cudaCreateChannelDesc<uchar4>();
    CSC(cudaMallocArray(&arr, &ch, w, h));
    CSC(cudaMemcpy2DToArray(arr, 0, 0, data, w * sizeof(uchar4), w * sizeof(uchar4), h, cudaMemcpyHostToDevice));

    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = arr;

    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = false;

    cudaTextureObject_t texObj = 0;
    CSC(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));


    uchar4 *dev_out;
    CSC(cudaMalloc(&dev_out, sizeof(uchar4) * w * h));



    kernel<<< dim3(16, 16), dim3(32, 32) >>>(texObj, dev_out, w, h);
    CSC(cudaGetLastError());

    CSC(cudaMemcpy(data, dev_out, sizeof(uchar4) * w * h, cudaMemcpyDeviceToHost));

    FILE *out;

    if ((out = fopen(out_name.c_str(), "wb")) == NULL) {
        cout << "File open error\n";
        return -1;
    }

    fwrite(&w, sizeof(int), 1, out);
    fwrite(&h, sizeof(int), 1, out);
    fwrite(data, sizeof(uchar4), w * h, out);


    fclose(in);
    fclose(out);

    cudaDestroyTextureObject(texObj);
    cudaFreeArray(arr);
    free(data);
    return 0;
}