#ifndef PROJECT_TENSOR_GPU_H
#define PROJECT_TENSOR_GPU_H

#include "gpu.h"
#include <string>
#include <iostream>

//struct t_shape;

struct cu_shape {
    int n_dims;
    int d1,d2,d3;
    int len;
    void print(std::string s);
    void print();
};

struct cu_tensor {
    float * array;
    cu_shape shape;
};


const cu_shape CU_SHAPE_DEF = {0,0,0,0,0};
const cu_tensor CU_TENSOR_DEF = {NULL, CU_SHAPE_DEF};

cu_shape new_cu_shape(int x);
cu_shape new_cu_shape(int x, int y);
cu_shape new_cu_shape(int x, int y, int z);
cu_tensor cu_alloc_tensor(cu_shape shape);
void cu_shape_print(cu_shape s, std::string caption);
void cu_shape_print(cu_shape s);
void cu_tensor_print(cu_tensor t, std::string caption);
void cu_tensor_print(cu_tensor t);

float w_find_max(cu_tensor t);
void w_reduce_sum(cu_tensor t, cu_tensor result);
void w_reduce_mean(cu_tensor t, cu_tensor result, int original_len);
void w_reduce_mean_axis(int axis, cu_tensor t, cu_tensor result, int original_len);
void w_tensor_fill(cu_tensor t, float val);

//cu_shape to_cu_shape(t_shape s);

#endif //PROJECT_TENSOR_GPU_H