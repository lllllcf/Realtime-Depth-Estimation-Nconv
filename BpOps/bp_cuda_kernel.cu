#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <math.h>
#include <stdio.h>
#include "bp_cuda.h"
#include <curand.h>
#include <curand_kernel.h>

namespace {

    template<typename scalar_t>
    __global__ void
    dist_kernel4_v2(const scalar_t *__restrict__ Pc, scalar_t *__restrict__ IPCnum,
                    long *__restrict__ args, int B, int Cc, int N, int M, int numfw, int H, int W) {
        int num = 1; // NUM
        int h = threadIdx.x + blockIdx.x * blockDim.x;
        int w = threadIdx.y + blockIdx.y * blockDim.y;
        int b = threadIdx.z + blockIdx.z * blockDim.z;
        
        if ((h < H) && (w < W) && (b < B)) {
            
            scalar_t best[1]; // NUM
            long argbest[1]; // NUM
            scalar_t bestX[1]; scalar_t bestY[1]; // NUM
#pragma unroll
            for (int i = 0; i < num; i++) {
                best[i] = 1e20;
                argbest[i] = 0;
                bestX[i] = 0; bestY[i] = 0;
            }
            int posX = w; int posY = h; int bestId = 0;
            for (int k = 1; k < max(H,W); k++){
                posX -= 1;
                
                for (int j = 0; j < num; j ++){
                    int vX = 1; int vY = 1;
                    if (j >= 0) vX = -1; // NUM
                    if (j == 0 || j == num - 1) vY = -1;
                    for (int i = 0; i < k; i ++){
                        posX += vX;
                        posY += vY;
                        if ((posX >= 0) && (posY >= 0) && (posX < W) && (posY < H)){
                            if (Pc[b * W * H + posY * W + posX] > 0.){
                                best[bestId] = k;
                                argbest[bestId] = W * posY + posX;//(long)(posY * W + posX);
                                bestX[bestId] = posX;
                                bestY[bestId] = posY;                                
                                bestId ++;
                                if (bestId >= num) break;
                            }
                        }
                    }
                    if (bestId >= num) break;
                }
                if (bestId >= num) break;
            }
#pragma unroll
            for (int ii = 0; ii < num; ii++) {
                
                //if (best[i] >= 1e20) {
                //    argbest[i] = argbest[i - 1];
                //}
                //args[b * num * N + i * N + h * W + w] = argbest[i];
                args[b * num * N + ii * N + h * W + w] = argbest[ii];
                // Ignored IPCnum. Maybe will regret it later
                IPCnum[b * Cc * num * N + 0 * num * N + ii * N + h * W + w] =
                        bestX[ii];
                IPCnum[b * Cc * num * N + 1 * num * N + ii * N + h * W + w] =
                        bestY[ii];
            }
            
         
        }
    }

    template<typename scalar_t>
    __global__ void
    dist_kernel4(const scalar_t *__restrict__ Pc, scalar_t *__restrict__ IPCnum,
                    long *__restrict__ args, int B, int Cc, int N, int M, int numfw, int H, int W) {
        int num = 1; // NUM
        int h = threadIdx.x + blockIdx.x * blockDim.x;
        int w = threadIdx.y + blockIdx.y * blockDim.y;
        int b = threadIdx.z + blockIdx.z * blockDim.z;
        //args[b * num * N + 0 * N + h * W + w] = num;
        
        if ((h < H) && (w < W) && (b < B)) {
            scalar_t best[1]; // NUM
            long argbest[1]; // NUM
#pragma unroll
            for (int i = 0; i < num; i++) {
                best[i] = 1e20;
                argbest[i] = 0;
            }
            for (int m = 0; m < M; m++) {
                scalar_t res1, res2, dist;
                res1 = Pc[b * Cc * M + 0 * M + m] - w;
                res2 = Pc[b * Cc * M + 1 * M + m] - h;
                //dist = res1 * res1 + res2 * res2;
                dist = abs(res1) + abs(res2);
#pragma unroll
                for (int i = 0; i < num; i++) {
                    if (best[i] >= dist) {
#pragma unroll
                        for (int j = num - 1; j > i; j--) {
                            best[j] = best[j - 1];
                            argbest[j] = argbest[j - 1];
                        }
                        best[i] = dist;
                        argbest[i] = m;
                        break;
                    }
                }
                //if (best[3] <= 10) break;
            }
#pragma unroll
            for (int i = 0; i < num; i++) {
                
                if (best[i] >= 1e20) {
                    argbest[i] = argbest[i - 1];
                }
                args[b * num * N + i * N + h * W + w] = argbest[i];
                IPCnum[b * Cc * num * N + 0 * num * N + i * N + h * W + w] =
                        Pc[b * Cc * M + 0 * M + argbest[i]] - w;
                IPCnum[b * Cc * num * N + 1 * num * N + i * N + h * W + w] =
                        Pc[b * Cc * M + 1 * M + argbest[i]] - h;
            }
         
        }
    }

    template<typename scalar_t>
    __global__ void
    conv2d_kernel_lf(scalar_t *__restrict__ x, scalar_t *__restrict__ y, scalar_t *__restrict__ z, size_t N1,
                     size_t N2, size_t Ci, size_t Co, size_t B,
                     size_t K) {
        int col_index = threadIdx.x + blockIdx.x * blockDim.x;
        int row_index = threadIdx.y + blockIdx.y * blockDim.y;
        int cha_index = threadIdx.z + blockIdx.z * blockDim.z;
        if ((row_index < N1) && (col_index < N2) && (cha_index < Co)) {
            for (int b = 0; b < B; b++) {
                scalar_t result = 0;
                for (int i = -int((K - 1) / 2.); i < (K + 1) / 2.; i++) {
                    for (int j = -int((K - 1) / 2.); j < (K + 1) / 2.; j++) {

                        if ((row_index + i < 0) || (row_index + i >= N1) || (col_index + j < 0) ||
                            (col_index + j >= N2)) {
                            continue;
                        }

                        result += x[b * N1 * N2 * Ci + cha_index * N1 * N2 + (row_index + i) * N2 + col_index + j] *
                                  y[b * N1 * N2 * Ci * K * K + cha_index * N1 * N2 * K * K +
                                    (i + (K - 1) / 2) * K * N1 * N2 +
                                    (j + (K - 1) / 2) * N1 * N2 + row_index * N2 + col_index];
                    }
                }
                z[b * N1 * N2 * Co + cha_index * N1 * N2 + row_index * N2 + col_index] = result;
            }
        }
    }


    template<typename scalar_t>
    __global__ void conv2d_kernel_lb(scalar_t *__restrict__ x, scalar_t *__restrict__ y, scalar_t *__restrict__ gx,
                                     scalar_t *__restrict__ gy, scalar_t *__restrict__ gz, size_t N1, size_t N2,
                                     size_t Ci, size_t Co, size_t B,
                                     size_t K) {
        int col_index = threadIdx.x + blockIdx.x * blockDim.x;
        int row_index = threadIdx.y + blockIdx.y * blockDim.y;
        int cha_index = threadIdx.z + blockIdx.z * blockDim.z;
        if ((row_index < N1) && (col_index < N2) && (cha_index < Co)) {
            for (int b = 0; b < B; b++) {
                scalar_t result = 0;
                for (int i = -int((K - 1) / 2.); i < (K + 1) / 2.; i++) {
                    for (int j = -int((K - 1) / 2.); j < (K + 1) / 2.; j++) {

                        if ((row_index - i < 0) || (row_index - i >= N1) || (col_index - j < 0) ||
                            (col_index - j >= N2)) {
                            continue;
                        }
                        result += gz[b * N1 * N2 * Ci + cha_index * N1 * N2 + (row_index - i) * N2 + col_index - j
                                  ] *
                                  y[b * N1 * N2 * Ci * K * K + cha_index * N1 * N2 * K * K +
                                    (i + (K - 1) / 2) * K * N1 * N2 +
                                    (j + (K - 1) / 2) * N1 * N2 + (row_index - i) * N2 + col_index - j];
                        gy[b * N1 * N2 * Ci * K * K + cha_index * N1 * N2 * K * K +
                           (i + (K - 1) / 2) * K * N1 * N2 +
                           (j + (K - 1) / 2) * N1 * N2 + (row_index - i) * N2 + col_index - j] =
                                gz[b * N1 * N2 * Ci + cha_index * N1 * N2 + (row_index - i) * N2 + col_index - j
                                ] * x[b * N1 * N2 * Ci + cha_index * N1 * N2 + row_index * N2 + col_index];

                    }
                }
                gx[b * N1 * N2 * Co + cha_index * N1 * N2 + row_index * N2 + col_index] = result;
            }
        }
    }


}


void Dist_Cuda(at::Tensor Pc, at::Tensor IPCnum, at::Tensor args,
                 size_t B, size_t Cc, size_t N, size_t M, size_t num, size_t H, size_t W) {
    dim3 blockSize(32, 32, 1);
    dim3 gridSize((H + blockSize.x - 1) / blockSize.x, (W + blockSize.x - 1) / blockSize.x, B);
    switch (num) {
        case 1: //NUM
            AT_DISPATCH_FLOATING_TYPES(Pc.scalar_type(), "DistF1_Cuda", ([&] {
                dist_kernel4_v2<scalar_t> << < gridSize, blockSize >> > (
                    Pc.data_ptr<scalar_t>(), IPCnum.data_ptr<scalar_t>(), args.data_ptr<long>(),
                            B, Cc, N, M, num, H, W);
            }));
            break;
        case 5:
            AT_DISPATCH_FLOATING_TYPES(Pc.scalar_type(), "DistF1_Cuda", ([&] {
                dist_kernel4<scalar_t> << < gridSize, blockSize >> > (
                    Pc.data_ptr<scalar_t>(), IPCnum.data_ptr<scalar_t>(), args.data_ptr<long>(),
                            B, Cc, N, M, num, H, W);
            }));
            break;
        default:
            exit(-1);
    }
}

void Conv2d_LF_Cuda(at::Tensor x, at::Tensor y, at::Tensor z, size_t N1, size_t N2, size_t Ci, size_t Co, size_t B,
                    size_t K) {
    dim3 blockSize(32, 32, 1);
    dim3 gridSize((N2 + blockSize.x - 1) / blockSize.x, (N1 + blockSize.y - 1) / blockSize.y,
                  (Co + blockSize.z - 1) / blockSize.z);
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "Conv2d_LF", ([&] {
        conv2d_kernel_lf<scalar_t> << < gridSize, blockSize >> > (
            x.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(), z.data_ptr<scalar_t>(),
                    N1, N2, Ci, Co, B, K);
    }));
}


void
Conv2d_LB_Cuda(at::Tensor x, at::Tensor y, at::Tensor gx, at::Tensor gy, at::Tensor gz, size_t N1, size_t N2,
               size_t Ci,
               size_t Co, size_t B, size_t K) {
    dim3 blockSize(32, 32, 1);
    dim3 gridSize((N2 + blockSize.x - 1) / blockSize.x, (N1 + blockSize.y - 1) / blockSize.y,
                  (Co + blockSize.z - 1) / blockSize.z);
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "Conv2d_LB", ([&] {
        conv2d_kernel_lb<scalar_t> << < gridSize, blockSize >> > (
            x.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(),
                    gx.data_ptr<scalar_t>(), gy.data_ptr<scalar_t>(), gz.data_ptr<scalar_t>(),
                    N1, N2, Ci, Co, B, K);
    }));
}
