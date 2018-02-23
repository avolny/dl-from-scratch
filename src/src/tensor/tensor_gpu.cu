#include "tensor_gpu.h"
//#include "tensor.h"
#include <device_launch_parameters.h>
#include <cmath>

cu_shape new_cu_shape(int x) {
    cu_shape s;
    s.n_dims = 1;
    s.d1 = x;
    s.d2 = 0;
    s.d3 = 0;
    s.len = x;
    return s;
}

cu_shape new_cu_shape(int x, int y) {
    cu_shape s;
    s.n_dims = 2;
    s.d1 = x;
    s.d2 = y;
    s.d3 = 0;
    s.len = x*y;
    return s;
}

cu_shape new_cu_shape(int x, int y, int z) {
    cu_shape s;
    s.n_dims = 3;
    s.d1 = x;
    s.d2 = y;
    s.d3 = z;
    s.len = x*y*z;
    return s;
}

cu_tensor cu_alloc_tensor(cu_shape shape) {
    cu_tensor t;
    t.shape = shape;
    CHECK_ERROR( cudaMalloc(&t.array, sizeof(float)*t.shape.len) );
    return t;
}


void cu_shape_print(cu_shape s, std::string caption) {
    std::cout << caption << " len " << s.len << " n_dims " << s.n_dims << " ";
    if(s.n_dims == 1)
        std::cout << "(" << s.d1 << ")" << std::endl;
    else if(s.n_dims == 2)
        std::cout << "(" << s.d1 << "," << s.d2 << ")" << std::endl;
    else if(s.n_dims == 3)
        std::cout << "(" << s.d1 << "," << s.d2 << "," << s.d3 << ")" << std::endl;
}

void cu_shape_print(cu_shape s) {
    cu_shape_print(s,"");
}


#define BLOCK_SIZE 1024


__global__ void k_tensor_fill(cu_tensor t, float val) {
    int glob_ix = blockIdx.x*blockDim.x + threadIdx.x;
    if(glob_ix < t.shape.len)
        t.array[glob_ix] = val;
}


void w_tensor_fill(cu_tensor t, float val) {
    int gridDim = (t.shape.len + BLOCK_SIZE - 1) / BLOCK_SIZE; //ceil

    k_tensor_fill<<<gridDim,BLOCK_SIZE>>>(t, val);
}


__global__ void k_reduce_mean_axis(int axis, cu_tensor t, cu_tensor _sum, int original_len) {
    __shared__ float cache[BLOCK_SIZE];

    int th_ix;
    int glob_ix;
    int glob_col,glob_row;
    bool th_in_bounds;
    int glob_on_axis;
    int glob_axis_size;

    if(axis == 0) {
        th_ix = threadIdx.y;
        glob_row = th_ix + blockIdx.y*blockDim.y;
        glob_ix = (blockIdx.y*blockDim.y + threadIdx.y) * t.shape.d2 + blockIdx.x;
        glob_on_axis = glob_row;
        glob_axis_size = t.shape.d1;

        if(glob_on_axis < glob_axis_size) {
            cache[th_ix] = t.array[glob_ix];
        }
    } else {
        th_ix = threadIdx.x;
        glob_col = th_ix + blockIdx.x*blockDim.x;
        glob_ix = blockIdx.y * t.shape.d2 + blockDim.x*blockIdx.x + threadIdx.x;
        glob_on_axis = glob_col;
        glob_axis_size = t.shape.d2;

        if(glob_on_axis < glob_axis_size) {
            cache[th_ix] = t.array[glob_ix];
        }
    }



    __syncthreads();


    int prev_len = axis == 0 ? blockDim.y : blockDim.x;
    int step = (prev_len + 1) / 2;
    while( true ) {
        if (th_ix + step < prev_len && glob_on_axis + step < glob_axis_size) {
            cache[th_ix] += cache[th_ix + step];
        }

        __syncthreads();

        if(step == 1)
            break;

        prev_len = step;
        step = (prev_len + 1) / 2;
    }


    if(th_ix == 0 && glob_on_axis < glob_axis_size) {
        int res_ix = blockIdx.y*gridDim.x + blockIdx.x;
        if(axis == 0) {
            if(gridDim.y == 1) // last reduce
                _sum.array[res_ix] = cache[0]/original_len;
            else
                _sum.array[res_ix] = cache[0];
        } else {
            if(gridDim.x == 1) // last reduce
                _sum.array[res_ix] = cache[0]/original_len;
            else
                _sum.array[res_ix] = cache[0];
        }
    }
}



void w_reduce_mean_axis(int axis, cu_tensor t, cu_tensor result, int original_len) {
    cu_tensor res = t;

    int i = 0;
    int onaxisdim = axis==0 ? res.shape.d1 : res.shape.d2;
    int offaxisdim = axis==0 ? res.shape.d2 : res.shape.d1;
    while(onaxisdim > 1) {
//        std::cout << "i " << i << " onaxisdim " << onaxisdim << std::endl;
        int grid_along = (int)std::ceil(onaxisdim*1./BLOCK_SIZE);
//        std::cout << "grid along " << grid_along << std::endl;

        dim3 gridDims;
        dim3 blockDims;
        cu_tensor _sum;
        if(axis == 0) {
            gridDims.x = res.shape.d2;
            gridDims.y = grid_along;
            gridDims.z = 1;
            blockDims.x = 1;
            blockDims.y = BLOCK_SIZE;
            blockDims.z = 1;
            if(grid_along > 1)
                _sum = cu_alloc_tensor(new_cu_shape(grid_along, res.shape.d2));
//            std::cout << "gridDims " << gridDims.x << " " << gridDims.y << std::endl;
        } else {
            gridDims.x = grid_along;
            gridDims.y = res.shape.d1;
            gridDims.z = 1;
            blockDims.x = BLOCK_SIZE;
            blockDims.y = 1;
            blockDims.z = 1;
            if(grid_along > 1)
                _sum = cu_alloc_tensor(new_cu_shape(res.shape.d1, grid_along));
        }

        if (grid_along == 1) // for the last iteration provide the result array
            k_reduce_mean_axis<<<gridDims, blockDims>>>(axis, res, result, original_len);
        else
            k_reduce_mean_axis<<<gridDims, blockDims>>>(axis, res, _sum, original_len);

        cudaDeviceSynchronize();
        if (i > 0) {
            // free every intermediate array, except the input one
            CHECK_ERROR( cudaFree(res.array) );
        }

        if (grid_along == 1)
            res = result;
        else
            res = _sum;

        onaxisdim = axis==0 ? res.shape.d1 : res.shape.d2;
        i++;
    }
}


__global__ void k_reduce_mean(cu_tensor t, cu_tensor _sum, int original_len) {
    __shared__ float cache[BLOCK_SIZE];

    int th_ix = threadIdx.x;
    int glob_ix = blockIdx.x*blockDim.x + threadIdx.x;

    if(glob_ix < t.shape.len) {
        cache[th_ix] = t.array[glob_ix];
    }

    __syncthreads();


    int prev_len = blockDim.x;
    int step = (prev_len + 1) / 2;
    while( true ) {
        if (th_ix + step < prev_len && glob_ix < t.shape.len)
            cache[th_ix] += cache[th_ix + step];

        __syncthreads();

        if(step == 1)
            break;

        prev_len = step;
        step = (prev_len + 1) / 2;
    }

    if(th_ix == 0 && glob_ix < t.shape.len) {
        // write the reduced sum to the array
        if(gridDim.x == 1) {
            _sum.array[blockIdx.x] = cache[0]/original_len; // divide by the len only in the last step for numerical stability
        } else {
            _sum.array[blockIdx.x] = cache[0];
        }
    }
}


void w_reduce_sum(cu_tensor t, cu_tensor result) {
    w_reduce_mean(t, result, 1);
}


void w_reduce_mean(cu_tensor t, cu_tensor result, int original_len) {
    cu_tensor res = t;

    int i = 0;
    while(res.shape.len > 1) {
        int grid_len = (int)std::ceil(res.shape.len*1./BLOCK_SIZE);
        cu_tensor _sum;
        if(grid_len != 1)
            _sum = cu_alloc_tensor(new_cu_shape(grid_len));

        if(grid_len == 1)
            k_reduce_mean<<<grid_len, BLOCK_SIZE>>>(res, result, original_len);
        else
            k_reduce_mean<<<grid_len, BLOCK_SIZE>>>(res, _sum, original_len);

        cudaDeviceSynchronize();
        if (i > 0) {
            // free every intermediate array, except the input one
            CHECK_ERROR( cudaFree(res.array) );
        }

        if(grid_len == 1) {
            break;
        }

        res = _sum;

        i++;
    }
}




__global__ void k_find_max(cu_tensor t, cu_tensor max_arr) {
    __shared__ float cache[BLOCK_SIZE*2];

    int glob_ix = blockIdx.x * blockDim.x * 2 + threadIdx.x * 2;
    int cacheIndex = threadIdx.x;

    if(cacheIndex < t.shape.len && glob_ix < t.shape.len) {
        cache[threadIdx.x * 2] = t.array[blockIdx.x * blockDim.x * 2 + threadIdx.x * 2];
        if(glob_ix+1 < t.shape.len) {
            cache[threadIdx.x * 2 + 1] = t.array[blockIdx.x * blockDim.x * 2 + threadIdx.x * 2 + 1];
        } else { //case with odd number of elements
            cache[threadIdx.x * 2 + 1] = cache[threadIdx.x * 2];
        }
    }

    __syncthreads();

    int step = blockDim.x;
    while ( step != 0 ) {
        if (cacheIndex < step && glob_ix < t.shape.len)
            cache[cacheIndex] = fmaxf(cache[cacheIndex],cache[cacheIndex + step]);
        __syncthreads(); // synchronizace vláken po provedení každé fáze
        step /= 2; // zmenšení kroku pro další fázi redukce
    }

    if(cacheIndex == 0 && glob_ix < t.shape.len) {
        max_arr.array[blockIdx.x] = cache[0];
    }
}

//find max of tensor using parallel reduction
float w_find_max(cu_tensor t) {
    int block_size = BLOCK_SIZE;
    cu_tensor t1 = t;
    cu_tensor max_array;
    int grid_len;
    int i = 0;
    do {
        grid_len = (int) std::ceil(t1.shape.len * 1.0 / block_size / 2);

        max_array.shape = new_cu_shape(grid_len);
        CHECK_ERROR(cudaMalloc(&max_array.array, sizeof(float) * grid_len));

        k_find_max <<< grid_len, block_size >>> (t1, max_array);
        cudaDeviceSynchronize();

        if(i > 0)
            CHECK_ERROR(cudaFree(t1.array));
        t1 = max_array;
        i++;
    } while(grid_len > 1);

//    float *max_array_host = new float[max_array.shape.len];
//    CHECK_ERROR( cudaMemcpy(max_array_host, max_array.array, sizeof(float)*max_array.shape.len, cudaMemcpyDeviceToHost) );
//
//    // most of the times it's faster to do the max using loop than another reduction
//    float max = max_array_host[0];
//    for (int i = 0; i < max_array.shape.len; ++i) {
//        std::cout << i << ", " << max_array_host[i] << std::endl;
//        if(max_array_host[i] > max)
//            max = max_array_host[i];
//    }

    float *max = new float;
    CHECK_ERROR( cudaMemcpy(max, max_array.array, sizeof(float), cudaMemcpyDeviceToHost) );
    CHECK_ERROR(cudaFree(max_array.array));

    return *max;
}

#define PRINT_FIRST 20
#define PRINT_LAST 5

void cu_tensor_print(cu_tensor t, std::string caption) {
    std::cout << caption << "tensor, shape ";
    cu_shape_print(t.shape);

    float * data = new float[t.shape.len];
    CHECK_ERROR( cudaMemcpy(data, t.array, sizeof(float)*t.shape.len, cudaMemcpyDeviceToHost) );

    std::cout << "{ ";
    if(t.shape.len < PRINT_FIRST+PRINT_LAST)
        for (int i = 0; i < t.shape.len; ++i) {
            std::cout << data[i];
            if(i != t.shape.len-1)
                std::cout << ", ";
        }
    else {
        for (int i = 0; i < PRINT_FIRST; ++i) {
            std::cout << data[i] << ", ";
        }
        std::cout << "... ";

        for (int i = t.shape.len-PRINT_LAST-1; i < t.shape.len; ++i) {
            std::cout << data[i];
            if(i != t.shape.len-1)
                std::cout << ", ";
        }
    }

    std::cout << " }" << std::endl << std::endl;
    delete data;
}

void cu_tensor_print(cu_tensor t) {
    cu_tensor_print(t, "");
}
