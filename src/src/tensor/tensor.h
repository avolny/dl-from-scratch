//
// Created by adam on 8.11.17.
//

#ifndef PROJECT_TENSOR_H
#define PROJECT_TENSOR_H

#include <initializer_list>
#include <stdexcept>
#include <iostream>
#include <algorithm>
#include <memory>
#include "tensor_gpu.h"

using namespace std;

struct t_shape {
    int* dims;
    int n_dims;
    t_shape();
    t_shape(const t_shape &t);
    t_shape(int* dims, int n_dims);
    t_shape(const initializer_list<int> &il);
    ~t_shape();
    bool eq(t_shape s);
    bool eq(t_shape* s);
    bool eq(cu_shape s);
    int flat_size();
    int& operator[](int index);
    void print();
};

typedef float (*tensor_fun_unary)(float a);
typedef float (*tensor_fun_binary)(float a, float b);
typedef float (*tensor_fun_universal)(float a, void* p);
//typedef float (*)

struct tensor1d;
struct tensor2d;
struct tensor3d;
struct tensor;

typedef std::shared_ptr<tensor1d> tensor1d_ptr;
typedef std::shared_ptr<tensor2d> tensor2d_ptr;
typedef std::shared_ptr<tensor3d> tensor3d_ptr;
typedef std::shared_ptr<tensor> tensor_ptr;

tensor_ptr new_tensor(t_shape shape);
//tensor_ptr new_tensor(const initializer_list<int> &il);

/**
 * Tensor wrapper
 */
struct tensor{
    t_shape shape;
    /// Constructor of tensor with given shape
    /// \param shape
    tensor(t_shape shape) : shape(shape) {};
    /// fill whole tensor with single value
    /// \param value
    virtual void fill(float value);
    virtual void add(tensor_ptr t);
    virtual void sub(tensor_ptr t);
    /// Multiply two tensors element by element, same shape required
    /// \param t
    virtual void multiply(tensor_ptr t);
    /// Multiply the tensor by a constant
    virtual void multiply_const(float val);
    virtual tensor_ptr transpose() = 0;
    /// A dot product between two 2D tensors
    virtual tensor_ptr dot(tensor_ptr t) = 0;
    virtual float sum() = 0;
    /// Performs sum along given axis
    /// \param axis
    /// \return
    virtual tensor_ptr sum(int axis) = 0;
    virtual float mean() = 0;
    virtual float max() = 0;
    virtual float min() = 0;
    virtual tensor_ptr flatten() = 0;
    /// Repeats the tensor along given axis
    /// \param axis
    /// \param n_times
    /// \return
    virtual tensor_ptr repeat(int axis, int n_times) = 0;
    /// Creates deep copy of the tensor
    /// \return
    virtual tensor_ptr copy() = 0;
    virtual tensor_ptr reshape(t_shape s) = 0;
    /// Apply unary function to each element
    /// \param f
    virtual void apply_fun(tensor_fun_unary f) = 0;
    /// Apply binary function to each element, it takes on input elements of this tensor and tensor t
    /// \param t
    /// \param f
    virtual void apply_fun(tensor_ptr t, tensor_fun_binary f) = 0;
    /// Apply a function which takes arbitrary input to each element
    /// \param p
    /// \param f
    virtual void apply_fun(void* p, tensor_fun_universal f) = 0;
    /// Print the tensor out
    virtual void print() = 0;
    void print(string caption);
    /// Compares shape of two tensors
    /// \param t
    /// \return
    bool dim_eq(tensor_ptr t);
    void force_dim_eq(tensor_ptr t);
};

struct tensor1d : tensor{
    float* array;
    int rows;

    explicit tensor1d(int rows);
    explicit tensor1d(t_shape shape);
    ~tensor1d();
    tensor_ptr dot(tensor_ptr t);
    float sum();
    tensor_ptr sum(int axis);
    float mean();
    float max();
    float min();
    tensor_ptr repeat(int axis, int n_times);
    tensor_ptr copy();
    tensor_ptr flatten();
    tensor_ptr reshape(t_shape s);
    tensor_ptr transpose();
    void apply_fun(tensor_fun_unary f);
    void apply_fun(tensor_ptr t, tensor_fun_binary f);
    void apply_fun(void* p, tensor_fun_universal f);
    void print();
};


struct tensor2d : tensor{
    float** array;
    int rows, cols;
    explicit tensor2d(int rows, int cols);
    explicit tensor2d(t_shape shape);
    ~tensor2d();
    tensor_ptr dot(tensor_ptr t);
    float sum();
    tensor_ptr sum(int axis);
    float mean();
    float max();
    float min();
    tensor_ptr repeat(int axis, int n_times);
    tensor_ptr copy();
    tensor_ptr flatten();
    tensor_ptr reshape(t_shape s);
    tensor_ptr transpose();
    void apply_fun(tensor_fun_unary f);
    void apply_fun(tensor_ptr t, tensor_fun_binary f);
    void apply_fun(void* p, tensor_fun_universal f);
    void print();
};



//struct tensor3d : tensor{
//    float*** array;
//    int rows, cols, depth;
//    explicit tensor3d(int rows, int cols, int depth);
//    void fill(float value);
//    void add(tensor3d* t);
//    void dot(tensor3d* t);
//    tensor3d* copy();
//    tensor1d* flatten();
//    tensor2d* reshape2d(t_shape s);
//    tensor3d* reshape3d(t_shape s);
//    void print();
//    bool dim_eq(tensor3d* t);
//};

t_shape* from_cu_shape(cu_shape s);
cu_shape to_cu_shape(t_shape s);
cu_tensor cu_alloc_tensor(t_shape shape);
tensor_ptr tensor_from_cu(cu_tensor tensor);
cu_tensor tensor_to_cu(tensor_ptr tensor);

tensor_ptr tensor2d_range(int rows, int cols);
tensor_ptr tensor1d_range(int rows);

#endif //PROJECT_TENSOR_H
