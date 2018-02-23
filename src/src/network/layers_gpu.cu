#include <cmath>
#include <device_launch_parameters.h>
#include "layers_gpu.h"
#include <iostream>
#include <string>

#define TILE_WIDTH 32
#define EPS 1e-4
#define BLOCK_SIZE 1024

/*
 *                  KERNEL CROSSENTROPY ERROR COMPUTE ACTIVATION sum(-t*log(y))
 */


__global__ void k_crossentropy_compute_activation(cu_tensor input, cu_tensor target, cu_tensor loss, cu_tensor _sum) {
    __shared__ float cache[BLOCK_SIZE];

    int th_ix = threadIdx.x;
    int glob_ix = blockIdx.x*blockDim.x + threadIdx.x;

    if(glob_ix < input.shape.len) {
        float diff = input.array[glob_ix]*target.array[glob_ix];
        float loss_v;
        if(diff == 0) {
            loss_v = 0;
        } else {
            loss_v = -logf(diff + EPS);
        }
        cache[th_ix] = loss_v;
        loss.array[glob_ix] = loss_v; //write it right away to the loss array

    }

    __syncthreads();

    int prev_len = blockDim.x;
    int step = (prev_len + 1) / 2;
    while( true ) {
        if(th_ix + step < prev_len && glob_ix < input.shape.len)
            cache[th_ix] += cache[th_ix + step];

        __syncthreads();

        if(step == 1)
            break;

        prev_len = step;
        step = (prev_len + 1) / 2;
    }

    if(th_ix == 0 && glob_ix < input.shape.len) {
        // write the reduced sum to the array
        _sum.array[blockIdx.x] = cache[0];
    }
}



/*
 *                  WRAPPER CROSSENTROPY ERROR COMPUTE ACTIVATION
 */


void w_crossentropy_compute_activation(cu_tensor input, cu_tensor target, cu_tensor loss, cu_tensor loss_sum) {
    int blockDim = 1024;
    int gridDim = (input.shape.len+blockDim-1) / blockDim; //ceil

    cu_tensor _sum = cu_alloc_tensor(new_cu_shape(gridDim));

    // run the kernel
    k_crossentropy_compute_activation<<<gridDim, blockDim>>>(input, target, loss, _sum);
    cudaDeviceSynchronize();


    // reduce the sum of the remaining elements

    w_reduce_mean(_sum, loss_sum, input.shape.len);


    CHECK_ERROR( cudaFree(_sum.array) );
}

/*
 *                  KERNEL CROSSENTROPY ERROR COMPUTE GRAD WRT OUTPUT
 */

__global__ void k_crossentropy_compute_grad_error_wrt_output(cu_tensor input, cu_tensor target, cu_tensor loss_grad) {
    int th_ix = threadIdx.x;
    int glob_ix = blockIdx.x*blockDim.x + threadIdx.x;

    if(glob_ix < input.shape.len) {
        float diff = input.array[glob_ix]*target.array[glob_ix];
        float loss_v;
        if(diff == 0) {
            loss_v = 0;
        } else {
            loss_v = -1/(diff + EPS)/loss_grad.shape.len;
        }
//        printf("th_ix %d loss %f\n",th_ix,loss_v);
        loss_grad.array[glob_ix] = loss_v; //write it right away to the loss array
    }
}

/*
 *                  WRAPPER CROSSENTROPY ERROR COMPUTE GRAD WRT OUTPUT
 */

void w_crossentropy_compute_grad_error_wrt_output(cu_tensor input, cu_tensor target, cu_tensor loss_grad) {
    int blockDim = 1024;
    int gridDim = (input.shape.len+blockDim-1) / blockDim; //ceil

    // run the kernel
    k_crossentropy_compute_grad_error_wrt_output<<<gridDim, blockDim>>>(input, target, loss_grad);
}



/*
 *                  KERNEL MSE COMPUTE ACTIVATION sum(1/2 * (y - t)^2)
 */

__global__ void k_mse_compute_activation(cu_tensor input, cu_tensor target, cu_tensor loss, cu_tensor _sum) {
    __shared__ float cache[BLOCK_SIZE];

    int th_ix = threadIdx.x;
    int glob_ix = blockIdx.x*blockDim.x + threadIdx.x;

    if(glob_ix < input.shape.len) {
        float diff = input.array[glob_ix]-target.array[glob_ix];
        float loss_v = diff*diff*0.5; //compute the loss for given output and given batch sample
        cache[th_ix] = loss_v;
        loss.array[glob_ix] = loss_v; //write it right away to the loss array
    }

    __syncthreads();

    int prev_len = blockDim.x;
    int step = (prev_len + 1) / 2;
    while( true ) {
        if(th_ix + step < prev_len && glob_ix < input.shape.len)
            cache[th_ix] += cache[th_ix + step];

        __syncthreads();

        if(step == 1)
            break;

        prev_len = step;
        step = (prev_len + 1) / 2;
    }

    if(th_ix == 0 && glob_ix < input.shape.len) {
        // write the reduced sum to the array
        _sum.array[blockIdx.x] = cache[0];
    }
}

/*
 *                  WRAPPER MSE COMPUTE ACTIVATION
 */

void w_mse_compute_activation(cu_tensor input, cu_tensor target, cu_tensor loss, cu_tensor loss_sum) {
    int blockDim = 1024;
    int gridDim = (input.shape.len+blockDim-1) / blockDim; //ceil

    cu_tensor _sum = cu_alloc_tensor(new_cu_shape(gridDim));

    // run the kernel
    k_mse_compute_activation<<<gridDim, blockDim>>>(input, target, loss, _sum);
    cudaDeviceSynchronize();

    // reduce the sum of the remaining elements
    w_reduce_mean(_sum, loss_sum, input.shape.len);
    cudaDeviceSynchronize();

    CHECK_ERROR( cudaFree(_sum.array) );
}

/*
 *                  KERNEL MSE COMPUTE GRAD ERROR WRT OUTPUT
 */
__global__ void k_mse_compute_grad_error_wrt_output(cu_tensor input, cu_tensor target, cu_tensor loss_grad) {
    int th_ix = threadIdx.x;
    int glob_ix = blockIdx.x*blockDim.x + threadIdx.x;

    if(glob_ix < input.shape.len) {
        float diff = input.array[glob_ix]-target.array[glob_ix];
        loss_grad.array[glob_ix] = diff/loss_grad.shape.len; //write it right away to the loss array
    }

}

/*
 *                  WRAPPER MSE COMPUTE GRAD ERROR WRT OUTPUT
 */

void w_mse_compute_grad_error_wrt_output(cu_tensor input, cu_tensor target, cu_tensor loss_grad) {
    int blockDim = 1024;
    int gridDim = (input.shape.len+blockDim-1) / blockDim; //ceil

    // run the kernel
    k_mse_compute_grad_error_wrt_output<<<gridDim, blockDim>>>(input, target, loss_grad);
}



/*
 *                  KERNEL COMPUTE DENSE LAYER ACTIVATION (Wx + b)
 */

__global__ void k_dense_compute_activation(cu_tensor weights, cu_tensor bias, cu_tensor input, cu_tensor activation, int k_steps) {
    __shared__ float tileWeights[TILE_WIDTH][TILE_WIDTH+1];
    __shared__ float tileInput[TILE_WIDTH][TILE_WIDTH+1];
    __shared__ float tileBias[TILE_WIDTH];



    int middle_dim = weights.shape.d2;

    int block_size = blockDim.x*blockDim.y;
    int glob_row = blockIdx.y*blockDim.y + threadIdx.y;
    int glob_col = blockIdx.x*blockDim.x + threadIdx.x;
    int glob_ix = glob_row*activation.shape.d2 + glob_col;

    float cell_value;

    //only threads within bounds will compute
    if(glob_row < activation.shape.d1 && glob_col < activation.shape.d2) {
        // first column will load bias
        if (threadIdx.x == 0) {
            tileBias[threadIdx.y] = bias.array[glob_row];
        }
    }

    __syncthreads();

    if(glob_row < activation.shape.d1 && glob_col < activation.shape.d2) {
        // bias is the same for all elements in a row and we can use it to initialize the value
        cell_value = tileBias[threadIdx.y];
    }


    // iterate over tiles
    for (int k = 0; k < k_steps; ++k) {

        int weights_row = blockIdx.y * blockDim.y + threadIdx.y;
        int weights_col = k * TILE_WIDTH + threadIdx.x;

        int input_row = k * TILE_WIDTH + threadIdx.y;
        int input_col = blockIdx.x * blockDim.x + threadIdx.x;


        // load tile
        if (weights_row < weights.shape.d1 && weights_col < weights.shape.d2) {
            // if the thread lies within bounds of weights matrix, load value to tile
            int weights_ix = weights_row * middle_dim + weights_col;
            tileWeights[threadIdx.y][threadIdx.x] = weights.array[weights_ix];
        }


        if (input_row < input.shape.d1 && input_col < input.shape.d2) {
            // if the thread lies within bounds of input matrix, load value to tile
            int input_ix = input_row * input.shape.d2 + input_col;
            tileInput[threadIdx.y][threadIdx.x] = input.array[input_ix];
        }


        __syncthreads();


        if(glob_row < activation.shape.d1 && glob_col < activation.shape.d2) {
            for (int i = 0; i < TILE_WIDTH; ++i) {
                if(k*TILE_WIDTH + i < middle_dim)
                    cell_value += tileWeights[threadIdx.y][i] * tileInput[i][threadIdx.x];
                else
                    break;
            }
        }

        __syncthreads();
    }

    if(glob_row < activation.shape.d1 && glob_col < activation.shape.d2) {
        activation.array[glob_ix] = cell_value;
    }
}


/*
 *                  WRAPPER DENSE COMPUTE ACTIVATION
 */

void w_dense_compute_activation(cu_tensor weights, cu_tensor bias, cu_tensor input, cu_tensor activation) {
    dim3 block(TILE_WIDTH,TILE_WIDTH,1); //block dims

    int x = (int) std::ceil(activation.shape.d2*1./TILE_WIDTH);
    int y = (int) std::ceil(activation.shape.d1*1./TILE_WIDTH);

    dim3 grid(x,y,1); //grid dims

    int k_steps = (int) std::ceil(weights.shape.d2*1./TILE_WIDTH); // the intermediate dimension of matrix multiply

    k_dense_compute_activation<<<grid, block>>>(weights, bias, input, activation, k_steps);
}

/*
 *                  KERNEL DENSE COMPUTE GRAD ERROR WRT WEIGHTS
 */


__global__ void k_dense_compute_grad_error_wrt_weights(
        cu_tensor grad_err_wrt_activ, cu_tensor input, cu_tensor grad_err_wrt_weights, int k_steps) {

    __shared__ float tileGradActivity[TILE_WIDTH][TILE_WIDTH+1];
    __shared__ float tileInputTranspose[TILE_WIDTH][TILE_WIDTH+1];


    int middle_dim = grad_err_wrt_activ.shape.d2;


    int glob_row = blockIdx.y*blockDim.y + threadIdx.y;
    int glob_col = blockIdx.x*blockDim.x + threadIdx.x;
    int glob_ix = glob_row*grad_err_wrt_weights.shape.d2 + glob_col;

    float cell_value = 0;

    // iterate over tiles
    for (int k = 0; k < k_steps; ++k) {

        int grad_activ_row = blockIdx.y * TILE_WIDTH + threadIdx.y;
        int grad_activ_col = k * TILE_WIDTH + threadIdx.x;

        // get offsets of the transposed tiles
        int input_col = k * TILE_WIDTH + threadIdx.x;
        int input_row = blockIdx.x * TILE_WIDTH + threadIdx.y;


        // load tile
        if (grad_activ_row < grad_err_wrt_activ.shape.d1 && grad_activ_col < grad_err_wrt_activ.shape.d2) {
            // if the thread lies within bounds of grad_activ matrix, load value to tile
            int grad_activ_ix = grad_activ_row * grad_err_wrt_activ.shape.d2 + grad_activ_col;
            tileGradActivity[threadIdx.y][threadIdx.x] = grad_err_wrt_activ.array[grad_activ_ix];
        }


        if (input_row < input.shape.d1 && input_col < input.shape.d2) {
            // if the thread lies within bounds of input matrix, load value to tile
            int input_ix = input_row * input.shape.d2 + input_col;
            tileInputTranspose[threadIdx.x][threadIdx.y] = input.array[input_ix]; //switched coords for transpose
        }


        __syncthreads();


        if(glob_row < grad_err_wrt_weights.shape.d1 && glob_col < grad_err_wrt_weights.shape.d2) {
            int k_tile = k*TILE_WIDTH;
            for (int i = 0; i < TILE_WIDTH; ++i) {
                if(k_tile + i < middle_dim)
                    // we transposed the second cache already during loading to cache
                    cell_value += tileGradActivity[threadIdx.y][i] * tileInputTranspose[i][threadIdx.x];
                else
                    break;
            }
        }

        __syncthreads();
    }

    if(glob_row < grad_err_wrt_weights.shape.d1 && glob_col < grad_err_wrt_weights.shape.d2) {
        grad_err_wrt_weights.array[glob_ix] = cell_value;
    }
}

/*
 *                  WRAPPER DENSE COMPUTE GRAD ERROR WRT WEIGHTS
 */

void w_dense_compute_grad_error_wrt_weights(cu_tensor grad_err_wrt_activ, cu_tensor input, cu_tensor grad_err_wrt_weights) {
    dim3 block(TILE_WIDTH,TILE_WIDTH,1); //block dims

    int x = (int) std::ceil(grad_err_wrt_weights.shape.d2*1./TILE_WIDTH);
    int y = (int) std::ceil(grad_err_wrt_weights.shape.d1*1./TILE_WIDTH);

    dim3 grid(x,y,1); //grid dims

    int k_steps = (int) std::ceil(grad_err_wrt_activ.shape.d2*1./TILE_WIDTH); // the intermediate dimension of matrix multiply

    k_dense_compute_grad_error_wrt_weights<<<grid, block>>>(grad_err_wrt_activ, input, grad_err_wrt_weights, k_steps);
}

/*
 *                  WRAPPER DENSE COMPUTE GRAD ERROR WRT BIAS
 */

void w_dense_compute_grad_error_wrt_bias(cu_tensor grad_err_wrt_activ, cu_tensor grad_err_wrt_bias) {
    w_reduce_mean_axis(1, grad_err_wrt_activ, grad_err_wrt_bias, 1);
}

/*
 *                  KERNEL DENSE COMPUTE GRAD ERROR WRT OUTPUT
 */

__global__ void k_dense_compute_grad_error_wrt_output(cu_tensor weights, cu_tensor grad_err_wrt_activ, cu_tensor grad_err_wrt_output, int k_steps) {

    __shared__ float tileWeightsTranspose[TILE_WIDTH][TILE_WIDTH+1];
    __shared__ float tileGradActiv[TILE_WIDTH][TILE_WIDTH+1];


    int middle_dim = weights.shape.d1;

    int glob_row = blockIdx.y*blockDim.y + threadIdx.y;
    int glob_col = blockIdx.x*blockDim.x + threadIdx.x;
    int glob_ix = glob_row*grad_err_wrt_output.shape.d2 + glob_col;

    float cell_value;

    // iterate over tiles
    for (int k = 0; k < k_steps; ++k) {

        // switched coordinates for the transpose
        int weights_col = blockIdx.y * TILE_WIDTH + threadIdx.x;
        int weights_row = k * TILE_WIDTH + threadIdx.y;

        int grad_activ_row = k * TILE_WIDTH + threadIdx.y;
        int grad_activ_col = blockIdx.x * TILE_WIDTH + threadIdx.x;

        // load tile
        if (weights_row < weights.shape.d1 && weights_col < weights.shape.d2) {
            // if the thread lies within bounds of weights matrix, load value to tile
            int weights_ix = weights_row * weights.shape.d2 + weights_col;
            tileWeightsTranspose[threadIdx.x][threadIdx.y] = weights.array[weights_ix]; //switched coords for transpose
        }


        if (grad_activ_row < grad_err_wrt_activ.shape.d1 && grad_activ_col < grad_err_wrt_activ.shape.d2) {
            // if the thread lies within bounds of grad_activ matrix, load value to tile
            int grad_activ_ix = grad_activ_row * grad_err_wrt_activ.shape.d2 + grad_activ_col;
            tileGradActiv[threadIdx.y][threadIdx.x] = grad_err_wrt_activ.array[grad_activ_ix];
        }


        __syncthreads();


        if(glob_row < grad_err_wrt_output.shape.d1 && glob_col < grad_err_wrt_output.shape.d2) {
            int k_tile = k*TILE_WIDTH;
            for (int i = 0; i < TILE_WIDTH; ++i) {
                if(k_tile + i < middle_dim)
                    // we transposed the second cache already during loading to cache
                    cell_value += tileWeightsTranspose[threadIdx.y][i] * tileGradActiv[i][threadIdx.x];
                else
                    break;
            }
        }

        __syncthreads();
    }

    if(glob_row < grad_err_wrt_output.shape.d1 && glob_col < grad_err_wrt_output.shape.d2) {
        grad_err_wrt_output.array[glob_ix] = cell_value;
    }
}

/*
 *                  WRAPPER DENSE COMPUTE GRAD ERROR WRT OUTPUT
 */

void w_dense_compute_grad_error_wrt_output(cu_tensor weights, cu_tensor grad_err_wrt_activ, cu_tensor grad_err_wrt_output) {
    dim3 block(TILE_WIDTH, TILE_WIDTH, 1); //block dims

    int x = (int) std::ceil(grad_err_wrt_output.shape.d2 * 1. / TILE_WIDTH);
    int y = (int) std::ceil(grad_err_wrt_output.shape.d1 * 1. / TILE_WIDTH);

    dim3 grid(x, y, 1); //grid dims

    int k_steps = (int) std::ceil(grad_err_wrt_activ.shape.d1 * 1. / TILE_WIDTH); // the intermediate dimension of matrix multiply

    k_dense_compute_grad_error_wrt_output << < grid, block >> >
                                                      (weights, grad_err_wrt_activ, grad_err_wrt_output, k_steps);

}