#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <string>
#include <unistd.h>

using namespace std;

#define CSC(call) \
do { \
    cudaError_t status = call; \
    if (status != cudaSuccess) { \
        fprintf(stderr, "ERROR is %s:%d. Message: %s\n", __FILE__, __LINE__, cudaGetErrorString(status)); \
        exit(0); \
    } \
} while(0)

__constant__ double3 constM [32];

__global__ void kernel(uchar4 *out, int n, int mcount) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int offset = blockDim.x * gridDim.x;

    while (idx < n) {

        if (mcount == 0) {
            break;
        }

        double max = 255 * 3;

        for (int i = 0; i < mcount; ++i) {

            double d = -((out[idx].x - (double)constM[i].x) * (out[idx].x - (double)constM[i].x) + 
                       (out[idx].y - (double)constM[i].y) * (out[idx].y - (double)constM[i].y) + 
                       (out[idx].z - (double)constM[i].z) * (out[idx].z - (double)constM[i].z));

            if (d > max || i == 0) {
                max = d;
                out[idx].w = i;
            }
        }

        idx += offset;
    }
}

bool operator!=(const double3 &r, const double3 &l) {
    return (abs(r.x - l.x) > 1e-4 || abs(r.y - l.y) > 1e-4 || abs(r.z - l.z) > 1e-4);
}


int main() {

	string in_name, out_name;

    int nc, w, h;

    cin >> in_name >> out_name;

    FILE *in;

    if ((in = fopen(in_name.c_str(), "rb")) == NULL) {
        cout << "File open error\n";
        return -1;
    }

    fread(&w, sizeof(int), 1, in);
    fread(&h, sizeof(int), 1, in);

    cin >> nc;

    double3 *m = (double3 *)malloc(sizeof(double3) * nc);

    uchar4 *data = (uchar4 *)malloc(sizeof(uchar4) * w * h);
    fread(data, sizeof(uchar4), w * h, in);

    fclose(in);

    for (int i = 0; i < nc; ++i) { 

        unsigned long long x, y;
        cin >> x >> y;

        m[i].x = data[y * w + x].x;
        m[i].y = data[y * w + x].y;
        m[i].z = data[y * w + x].z;
    }


    bool end = true;

    uchar4 *dev_data;
    CSC(cudaMalloc(&dev_data, sizeof(uchar4) * w * h));

    do {
        if (nc == 0) {
            break;
        }

        CSC(cudaMemcpy(dev_data, data, sizeof(uchar4) * w * h, cudaMemcpyHostToDevice));
        CSC(cudaMemcpyToSymbol(constM, m, sizeof(double3) * nc, 0, cudaMemcpyHostToDevice));

        kernel<<<1024, 1024>>>(dev_data, w * h, nc);

        CSC(cudaDeviceSynchronize());
        CSC(cudaGetLastError());
        
        CSC(cudaMemcpy(data, dev_data, sizeof(uchar4) * w * h, cudaMemcpyDeviceToHost));

        vector<unsigned long long> count(nc);
        vector<double3> sum(nc);

        for (int i = 0; i < w * h; ++i) {

            count[data[i].w] += 1;

            sum[data[i].w].x += data[i].x;
            sum[data[i].w].y += data[i].y;
            sum[data[i].w].z += data[i].z;
        }


        end = false;

        for (int i = 0; i < nc; ++i) {
            sum[i].x = sum[i].x / count[i];
            sum[i].y = sum[i].y / count[i];
            sum[i].z = sum[i].z / count[i];

            if (m[i] != sum[i]) {
                end = true;
                m[i].x = sum[i].x;
                m[i].y = sum[i].y;
                m[i].z = sum[i].z;
            }
        }
    } while(end);

    
    FILE *out;

    if ((out = fopen(out_name.c_str(), "wb")) == NULL) {
        cout << "File open error\n";
        return -1;
    }

    fwrite(&w, sizeof(int), 1, out);
    fwrite(&h, sizeof(int), 1, out);
    fwrite(data, sizeof(uchar4), w * h, out);

    fclose(out);

    CSC(cudaFree(dev_data));
    free(data);
    free(m);
	return 0;
}
