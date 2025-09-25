#pragma once

#include <stdexcept>
#include <cuda_runtime.h>

#include "buffers.cuh"

void launch_yuyv2rgb_chw(uint32_t* yuyv, float* rgb, size_t length, int height, int width, int n_blocks, int block_dim, cudaStream_t stream = 0);

class Preprocessor {
    private:
        int block_dim;
        cudaStream_t stream;
        bool stream_created;

    public:
        Preprocessor(int _block_dim = 256) : block_dim(_block_dim), stream_created(false) {
            cudaError_t err = cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, -1);
            if (err == cudaSuccess) {
                stream_created = true;
            } else {
                stream = 0;
            }
        }

        ~Preprocessor() {
            if (stream_created) {
                cudaStreamDestroy(stream);
            }
        }

        void process(const GPUBuffer& buf, int height, int width) {
            const int n_blocks = static_cast<int>(((buf.length >> 2) + block_dim - 1) / block_dim);
            
            launch_yuyv2rgb_chw(reinterpret_cast<uint32_t*>(buf.data), buf.out, buf.length, height, width, n_blocks, block_dim, stream);
            
            cudaError_t err = cudaStreamSynchronize(stream);
            if (err != cudaSuccess){
                printf("CUDA error: %s\n", cudaGetErrorString(err));
                throw std::runtime_error("CUDA kernel execution failed");
            }
            
        }
        
};