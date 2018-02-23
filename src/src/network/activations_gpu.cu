#include "activations_gpu.h"
#include <cmath>
#include <device_launch_parameters.h>
#include <stdexcept>


/*
 *              COMPUTE ACTIVITY OF MOST ACTIVATIONS
 */

__global__ void k_activations_compute_activity(cu_tensor activation, cu_tensor activity, int function, cu_tensor* params, int n_params) {
    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    if(ix < activity.shape.len) {
        if (function == FUNCTION_RELU) {
            activity.array[ix] = fmaxf(activation.array[ix],0);
        } else if (function == FUNCTION_SIGMOID) {
            activity.array[ix] = 1/(1+expf(-activation.array[ix]));
        } else if (function == FUNCTION_LEAKY_RELU) {
            float val = activation.array[ix];
            if(val < 0) {
                activity.array[ix] = 0.1*val;
            } else {
                activity.array[ix] = val;
            }
        }
    }
}

/*
 *              COMPUTE ACTIVITY OF SOFTMAX
 */

__global__ void k_activations_compute_activity_softmax(cu_tensor activation, cu_tensor activity) {
    extern __shared__ float cache[];

    int th_ix = threadIdx.y;
    int glob_ix = threadIdx.y*activation.shape.d2 + blockIdx.x; //threadIdx gives row, block gives column
    int offset = blockDim.y; // offset in cache

    if(th_ix < activation.shape.d1) {
        float v = activation.array[glob_ix];
        cache[th_ix] = v;
        cache[th_ix + offset] = v;
    }

    __syncthreads();

    // find max in each column (block)
    int prev_len = (blockDim.y);
    int step = (prev_len + 1) / 2;
    while ( true ) {
        if (th_ix + step < prev_len && th_ix < activation.shape.d1)
            cache[th_ix + offset] = fmaxf(cache[th_ix + offset],cache[th_ix + offset + step]);
        __syncthreads(); // synchronizace vláken po provedení každé fáze

        if(prev_len == 1) // because of ceiling the step we have to break manually because it'll never reach 0
            break;

        prev_len = step;
        step = (step + 1) / 2; // zmenšení kroku pro další fázi redukce
    }


    // first subtract the maximum for numerical stability of the softmax
    float max = cache[0];
    cache[th_ix] -= max;
    cache[th_ix] = expf(cache[th_ix]); //apply the exponential
    cache[th_ix+offset] = cache[th_ix]; //copy value

    __syncthreads();

    // compute the sum of the exponentials
    prev_len = (blockDim.y);
    step = (prev_len + 1) / 2;
    while ( true ) {
        if (th_ix+step < prev_len && th_ix < activation.shape.d1)
            cache[th_ix + offset] += cache[th_ix + offset + step];
        __syncthreads(); // synchronizace vláken po provedení každé fáze

        if(prev_len == 1) // because of ceiling the step we have to break manually because it'll never reach 0
            break;

        prev_len = step;
        step = (step + 1) / 2; // zmenšení kroku pro další fázi redukce
    }

    __syncthreads();

    float sum = cache[offset]; //sum is stored in the first value of the offseted cache
    activity.array[glob_ix] = cache[th_ix]/sum; // divide the exponential by the sum of all others and write to global mem
}

/*
 *              COMPUTE ACTIVITY KERNELS WRAPPER
 */

void w_activations_compute_activity(cu_tensor activation, cu_tensor activity, int function, cu_tensor* params, int n_params) {
    int block_len = 1024;

    if(function == FUNCTION_SOFTMAX) {
        if(activation.shape.n_dims != 2) {
            cu_shape_print(activation.shape, "activation shape:");
            throw std::runtime_error("void w_activations_compute_activity CAN HANDLE ONLY 2D ARRAYS FOR NOW");
        }
        if(activation.shape.d1 > 1024)
            throw std::invalid_argument("this cuda implementation only supports softmax up to 1024 units");
        dim3 blockDim(1,activation.shape.d1);
        dim3 gridDim(activation.shape.d2);

        int shared_mem_size = activation.shape.d1*2; // half cache for
        k_activations_compute_activity_softmax<<<gridDim, blockDim, shared_mem_size>>>(activation, activity);
    } else {
        int grid_len = (int) std::ceil(activation.shape.len*1.0/block_len);

        k_activations_compute_activity<<<grid_len, block_len>>>(activation, activity, function, params, n_params);
    }
}


/////////////////////////////////////////////////////////////////////////////////////


/*
 *              COMPUTE GRADIENT OF MOST ACTIVATIONS
 */

__global__ void k_activations_compute_grad_wrt_activity(
        cu_tensor activation, cu_tensor activity, cu_tensor grad_err_wrt_output_prev,
        cu_tensor grad_output_wrt_activity, cu_tensor grad_err_wrt_activity ,int function, cu_tensor* params, int n_params) {
    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    if(ix < activation.shape.len) {
        float grad = 0;
        if (function == FUNCTION_RELU) {
            if(activation.array[ix] > 0)
                grad = 1;
            else
                grad = 0;
        } else if (function == FUNCTION_SIGMOID) {
            float y = activity.array[ix];
            grad = y*(1-y);
        } else if (function == FUNCTION_LEAKY_RELU) {
            if(activation.array[ix] < 0) {
                grad = 0.1;
            } else {
                grad = 1;
            }
        }
        grad_output_wrt_activity.array[ix] = grad; //the computed grad is just output wrt. activity
        grad_err_wrt_activity.array[ix] = grad_err_wrt_output_prev.array[ix]*grad; //here we compute err wrt. activity
    }
}

/*
 *              COMPUTE GRADIENT OF SOFTMAX
 */

__global__ void k_activations_compute_grad_wrt_activity_softmax(
        cu_tensor activation, cu_tensor activity, cu_tensor grad_err_wrt_output_prev,
        cu_tensor grad_output_wrt_activity, cu_tensor grad_err_wrt_activity) {
    extern __shared__ float cache[];

    int th_ix = threadIdx.y;
    int glob_ix = threadIdx.y*activation.shape.d2 + blockIdx.x;
    int offset = blockDim.y;

    float cell_activity;
    //cache the activity and previous layer gradient
    if(th_ix < activation.shape.d1) {
        cell_activity = activity.array[glob_ix];
        cache[th_ix] = cell_activity;
        cache[th_ix + offset] = grad_err_wrt_output_prev.array[glob_ix];
    }

    float cell_grad = 0;

    __syncthreads();

    if(th_ix < activation.shape.d1) {
        // first we incorporate the gradient of non-equal indices (-y_i*y_j)
        for (int j = 0; j < blockDim.y; ++j) {
            cell_grad += -cache[j + offset]*cache[j]*cell_activity;
        }

        //to avoid divergence we added one wrong gradient for j == th_ix
        cell_grad += cache[th_ix + offset]*cell_activity*cell_activity; //remove the faulty gradient for th_ix == j
        cell_grad += cache[th_ix + offset]*cell_activity*(1-cell_activity); //add the gradient for th_ix = j

        //assign the gradients
        grad_err_wrt_activity.array[glob_ix] = cell_grad;
//        grad_err_wrt_activity.array[glob_ix] = grad_err_wrt_output_prev.array[glob_ix]*cell_grad;
    }
}

/*
 *              COMPUTE GRADIENT KERNELS WRAPPER
 */

void w_activations_compute_grad_wrt_activity(cu_tensor activation, cu_tensor activity, cu_tensor grad_err_wrt_output_prev,
                                             cu_tensor grad_output_wrt_activity, cu_tensor grad_err_wrt_activity, int function, cu_tensor* params, int n_params) {
    int block_len = 1024;

    if(function == FUNCTION_SOFTMAX) {
        if(activation.shape.n_dims != 2) {
            cu_shape_print(activation.shape, "activation shape:");
            throw std::runtime_error("void w_activations_compute_grad_wrt_activity CAN HANDLE ONLY 2D ARRAYS FOR NOW");
        } if(activation.shape.d1 > 1024)
            throw std::invalid_argument("this cuda implementation only supports softmax up to 1024 units");

        dim3 blockDim(1,activation.shape.d1);
        dim3 gridDim(activation.shape.d2);

        int shared_mem_size = activation.shape.d1*2; // half cache for
        k_activations_compute_grad_wrt_activity_softmax<<<gridDim, blockDim, shared_mem_size>>>(
                activation, activity, grad_err_wrt_output_prev, grad_output_wrt_activity, grad_err_wrt_activity);
    } else {
        int grid_len = (int) std::ceil(activation.shape.len*1.0/block_len);

        k_activations_compute_grad_wrt_activity<<<grid_len, block_len>>>(
                activation, activity, grad_err_wrt_output_prev, grad_output_wrt_activity, grad_err_wrt_activity,
                        function, params, n_params);
    }
}