// based off from example from https://docs.nvidia.com/cuda/cuda-c-programming-guide/
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// one over PCIe/NVlink
__global__ void testTransfer(float *output, float *input_a, float *input_b) { //simple kernel to test our NCU visualizes data flow 
    int array_idx = (threadIdx.x *  pow(2, 15)) + blockIdx.x * (blockDim.x *  pow(2, 15));
    for(int i = array_idx; i < (array_idx +  pow(2, 15)); i++) {
        output[i] = (input_a[i] + input_b[i]) * 10; 
    }
} 

int main() {
    int tBlockCount = 64; //setting up host variables and input matrix
    int threadCount = 64;
    int rows = tBlockCount;
    int cols = threadCount * pow(2, 15);
    int array_size = (rows * cols * sizeof(float));

    float* host_output  = (float*)malloc(array_size);
    float* host_input_a = (float*)malloc(array_size);
    float* host_input_b = (float*)malloc(array_size);

    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            host_input_a[(i * j) + j] = (i * j) + j;
            host_input_b[(i * j) + j] = (i * j) + j;
        }
    }

    cudaSetDevice(0); //start setting up GPU 0
    cudaStream_t stream0;
    cudaStreamCreate(&stream0);

    int peer_avail; // enables peer memory access
    cudaDeviceCanAccessPeer(&peer_avail, 0 , 1);

    float* device_output; //allocate onto GPU 0
    float* device_input_a;
    float* device_input_b;
    cudaMalloc((void**)&device_output, array_size);
    cudaMalloc((void**)&device_input_a, array_size);
    cudaMalloc((void**)&device_input_b, array_size);
    
    cudaMemcpy(device_input_a, host_input_a, array_size, cudaMemcpyHostToDevice); //copy to Device 0
    cudaMemcpy(device_input_b, host_input_b, array_size, cudaMemcpyHostToDevice);

    testTransfer<<<tBlockCount, threadCount>>>(device_output, device_input_a, device_input_b);
    cudaDeviceSynchronize();
    cudaMemcpy(host_output, device_output, array_size, cudaMemcpyDeviceToHost); //writeback results to system memory





    cudaSetDevice(1); //switch over to other GPU
    cudaDeviceEnablePeerAccess(1, 0); //enable peer memory
    cudaStream_t stream1;
    cudaStreamCreate(&stream1);

    float* host_output_1  = (float*)malloc(array_size); 
    float* device_output_1;
    float* device_input_a_1;
    float* device_input_b_1;
    cudaMalloc((void**)&device_output_1, array_size);
    // cudaMemcpyPeerAsync(device_input_a_1, 1, device_input_a, 0, array_size, stream1);
    // cudaMemcpyPeerAsync(device_input_b_1, 1, device_input_b, 0, array_size, stream1);

    testTransfer<<<tBlockCount, threadCount>>>(device_output_1, device_input_a, device_input_b);
    cudaDeviceSynchronize();

    cudaMemcpy(host_output_1, device_output_1, array_size, cudaMemcpyDeviceToHost);

    cudaFree(device_output);
    cudaFree(device_input_a);
    cudaFree(device_input_b);

    // std::ofstream myfile;
    // myfile.open ("memory_test.txt");
    // for(int i = 0; i < rows; i++) {
    //     for(int j = 0; j < cols; j++) {
    //         myfile << host_output[(i * j) + j] << " ";
    //     }
    //     myfile << "\n";
    // }
    // myfile.close();

    cudaSetDevice(0);

    cudaFree(device_output);
    cudaFree(device_input_a);
    cudaFree(device_input_b);

    free(host_output);
    free(host_input_a);
    free(host_input_b);

    return 0;
}