#include <cuda_runtime.h>

#include "preprocessing.hpp"

#define clamp(x) (max(0, min(255, (x))))

__constant__ float norm_scale[3];
__constant__ float norm_offset[3];

__device__ inline float normalize(float x, int c) {
    return x * norm_scale[c] + norm_offset[c];
}

//yuyv2rgb Credits: https://stackoverflow.com/questions/72056909/convert-yuv2-yuyv-frames-to-rgb-without-use-of-opencv

constexpr int Y_OFFSET = 16;
constexpr int UV_OFFSET = 128;
constexpr int YUV2RGB_11 = 298;
constexpr int YUV2RGB_12 = -1;
constexpr int YUV2RGB_13 = 409;
constexpr int YUV2RGB_22 = -100;
constexpr int YUV2RGB_23 = -210;
constexpr int YUV2RGB_32 = 519;
constexpr int YUV2RGB_33 = 0;

__global__ void yuyv2rgb_chw(uint32_t* yuyv, float* rgb, size_t length, int height, int width){
    const int pair_idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int hw_pixels = height * width;
    const int total_pairs = length >> 2;

    if (pair_idx >= total_pairs) {
        return;
    }

    // YUYV -> RGB conversion

    const uint32_t pair = yuyv[pair_idx];
    const int Y0 = (pair & 0xFF) - Y_OFFSET;
    const int U = ((pair >> 8) & 0xFF) - UV_OFFSET;
    const int Y1 = ((pair >> 16) & 0xFF) - Y_OFFSET;
    const int V = ((pair >> 24) & 0xFF) - UV_OFFSET;

    const int uv_r = YUV2RGB_12 * U + YUV2RGB_13 * V;
    const int uv_g = YUV2RGB_22 * U + YUV2RGB_23 * V;
    const int uv_b = YUV2RGB_32 * U + YUV2RGB_33 * V;

    const int y0_scaled = YUV2RGB_11 * Y0;
    const int R0 = clamp((y0_scaled + uv_r) >> 8);
    const int G0 = clamp((y0_scaled + uv_g) >> 8);
    const int B0 = clamp((y0_scaled + uv_b) >> 8);

    const int y1_scaled = YUV2RGB_11 * Y1;
    const int R1 = clamp((y1_scaled + uv_r) >> 8);
    const int G1 = clamp((y1_scaled + uv_g) >> 8);
    const int B1 = clamp((y1_scaled + uv_b) >> 8);
    
    // Normalize

    const float R0_n = normalize(R0, 0);
    const float R1_n = normalize(R1, 0);

    const float G0_n = normalize(G0, 1);
    const float G1_n = normalize(G1, 1);

    const float B0_n = normalize(B0, 2);
    const float B1_n = normalize(B1, 2);

    //CHW conversion

    const int pixel_base = pair_idx << 1;
    const int R0_index = pixel_base;
    const int R1_index = pixel_base + 1;
    
    const int G0_index = R0_index + hw_pixels;
    const int G1_index = R1_index + hw_pixels;
    
    const int B0_index = G0_index + hw_pixels;
    const int B1_index = G1_index + hw_pixels;

    rgb[R0_index] = R0_n;
    rgb[R1_index] = R1_n;
    rgb[G0_index] = G0_n;
    rgb[G1_index] = G1_n;
    rgb[B0_index] = B0_n;
    rgb[B1_index] = B1_n;
}


static bool norm_params_set = false;
inline void ensure_normalization_params() {
    if (!norm_params_set) {
        // mean : r : 0.485, g : 0.456, g : 0.406
        // std : r : 0.229, g : 0.224, b : 0.225

        constexpr float mean[3] = {0.485f, 0.456f, 0.406f};
        constexpr float std[3]  = {0.229f, 0.224f, 0.225f};

        float scale[3], offset[3];
        for (int i = 0; i < 3; i++) {
            scale[i]  = 1.0f / (255.0f * std[i]);
            offset[i] = -mean[i] / std[i];
        }

        cudaMemcpyToSymbol(norm_scale, scale, sizeof(scale));
        cudaMemcpyToSymbol(norm_offset, offset, sizeof(offset));
        norm_params_set = true;
    }
}

void launch_yuyv2rgb_chw(uint32_t* yuyv, float* rgb, size_t length, int height, int width, int n_blocks, int block_dim, cudaStream_t stream) {
    ensure_normalization_params();
    const dim3 grid(static_cast<unsigned int>(n_blocks));
    const dim3 block(static_cast<unsigned int>(block_dim));
    yuyv2rgb_chw<<<grid, block, 0, stream>>>(yuyv, rgb, length, height, width);
}