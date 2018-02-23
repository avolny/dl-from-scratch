//
// Created by adam on 8.11.17.
//

#ifndef PROJECT_LAYERS_H
#define PROJECT_LAYERS_H

#include "tensor.h"
#include "activations.h"
#include <string>
#include <sstream>
#include <random>
#include "tensor_gpu.h"
#include "layers_gpu.h"

using namespace std;

class WeightInitializer;

/**
 * Core class of the framework, interface for each layer. Each layer can be seen as a node in the computation graph
 * as such, it has an input, it has to be capable of computing the output from the given input,
 * and it has to be able to compute it's parameter gradients and also the gradient of error w.r.t. the input of this layer,
 * which can then be backpropagated to previous layers.
 *
 * Clearly this doesn't hold for InputLayer.
 */
class Layer {
public:
    Layer(t_shape input_shape, t_shape output_shape, string name);
    Layer(string name);
    Layer();
    ~Layer();
    /// the method which is called during forward propagation
    virtual void compute_output();
    /// called during the back propagation
    virtual void compute_gradient();
    /// return the output tensor of this layer
    virtual tensor_ptr get_output() = 0;
    /// true if the layer is InputLayer
    virtual bool is_input() = 0;
    /// returns number of trainable tensors (for example weights, biases, etc.)
    virtual int get_num_trainable(); //returns number of types of trainable parameters
    /// shape of nth trainable tensor
    /// \param n
    /// \return
    virtual t_shape get_trainable_shape(int n); //returns shape of tensor of a given trainable type
    /// returns tensor of gradient of output of this layer w.r.t. the activation of this layer
    virtual tensor_ptr get_grad_output_wrt_activ();
    /// get tensor of gradient of error w.r.t. output of previous layer
    virtual tensor_ptr get_grad_error_wrt_output();
    /// get tensor of gradient of error w.r.t. nth parameter
    virtual tensor_ptr get_grad_error_wrt_param(int n); //returns gradient tensor for given parameter type
    /// get the nth parameter tensor
    virtual tensor_ptr get_trainable(int n);
    // CUDA VERSIONS OF ABOVE FUNCTIONS
    virtual cu_tensor cu_get_output() = 0;
    virtual cu_tensor cu_get_grad_output_wrt_activ();
    virtual cu_tensor cu_get_grad_error_wrt_output();
    virtual cu_tensor cu_get_grad_error_wrt_param(int n);
    virtual cu_tensor cu_get_trainable(int n);
    virtual cu_shape cu_get_trainable_shape(int n);
    virtual void set_input_shape(t_shape input_shape);
    virtual void set_output_shape(t_shape output_shape);
    /// initialize layer, its parameters
    virtual void initialize(int batch_size, bool use_gpu, bool use_cpu);
    /// resets computation flags
    virtual void reset_output_computation();
    /// resets gradient computation flags
    virtual void reset_grad_computation();
    /// add input layer
    virtual void add_parent(Layer* l);
    /// add succesor layer
    void add_child(Layer* l);
    bool computed_output = false;
    bool computed_grad = false;
    bool initialized = false;
    t_shape* input_shape;
    t_shape* output_shape;
protected:
    bool use_gpu;
    bool use_cpu;
    int batch_size;
    vector<Layer*> parents;
    vector<Layer*> children;
    string name;
};


class InputLayer : public Layer {
public:
    InputLayer(t_shape* input_shape);
    ~InputLayer();
    tensor_ptr get_output();
    cu_tensor cu_get_output();
    bool is_input();
    int get_num_trainable();
    void set_input(tensor_ptr input);
    void cu_set_input(cu_tensor input);
    void compute_output();
    void compute_gradient();
    void initialize(int batch_size, bool use_gpu, bool use_cpu);
    void reset_output_computation();
    void add_parent(Layer* l);
private:
    t_shape* data_shape;
    tensor_ptr batch;
    cu_tensor cu_batch;
    static int static_id;
};



class DenseLayer : public Layer {
public:
    DenseLayer(int n_outs, Activation::Function act, WeightInitializer* winit);
    ~DenseLayer();
    void compute_output();
    virtual void compute_gradient();
    tensor_ptr get_output();
    cu_tensor cu_get_output();
    bool is_input();
    int get_num_trainable();
    t_shape get_trainable_shape(int n);
    tensor_ptr get_trainable(int n);
    tensor_ptr get_grad_output_wrt_activ();
    tensor_ptr get_grad_error_wrt_output();
    tensor_ptr get_grad_error_wrt_param(int n);
    cu_tensor cu_get_grad_output_wrt_activ();
    cu_tensor cu_get_grad_error_wrt_output();
    cu_tensor cu_get_grad_error_wrt_param(int n);
    cu_tensor cu_get_trainable(int n);
    cu_shape cu_get_trainable_shape(int n);
    void initialize(int batch_size, bool use_gpu, bool use_cpu);
    void add_parent(Layer* l);
private:
    int n_outs;
    tensor_ptr activation_tensor;
    tensor_ptr weight_tensor;
    tensor_ptr bias_tensor;
    Activation::Function activation_function;
    Activation* activation;
    WeightInitializer* winit;
    static int static_id;

    tensor_ptr grad_error_wrt_output;
    tensor_ptr grad_error_wrt_weights;
    tensor_ptr grad_error_wrt_bias;

    cu_tensor cu_activation_tensor;
    cu_tensor cu_weight_tensor;
    cu_tensor cu_bias_tensor;
    cu_tensor cu_grad_error_wrt_output;
    cu_tensor cu_grad_error_wrt_weights;
    cu_tensor cu_grad_error_wrt_bias;



};

/**
 * Layer which lets you choose a loss function for the network.
 * It's output is a scalar loss (tensor2d 1x1)
 */
class LossLayer : public Layer {
public:
    /// Default constructor
    /// \param has_target true if the loss is trained in supervised manner
    LossLayer(bool has_target) : supervised(has_target) {};
    ~LossLayer();
    virtual bool has_target() {return supervised; };
    virtual bool is_input() {return false; };
    virtual void reset_output_computation();
    virtual tensor_ptr get_output() {return loss;} // returns loss value
    virtual tensor_ptr get_grad_error_wrt_output() {return loss_grad;};
    virtual void initialize(int batch_size, bool use_gpu, bool use_cpu);
    cu_tensor cu_get_output();
    cu_tensor cu_get_grad_error_wrt_output();
    virtual void cu_set_target(cu_tensor target);
    /// set target values for the network for supervised loss
    /// \param target tensor of desired outputs
    virtual void set_target(tensor_ptr target);
protected:
    bool supervised;
    tensor_ptr target;
    tensor_ptr loss;
    tensor_ptr loss_grad;
    cu_tensor cu_target;
    cu_tensor cu_loss;
    cu_tensor cu_loss_grad;
    cu_tensor cu_loss_sum;
};


class MSE : public LossLayer {
public:
    /// The Mean Square Error loss layer, it is supervised
    MSE() : LossLayer(true) {};
//    bool is_input() {return false;};
    void initialize(int batch_size, bool use_gpu, bool use_cpu);
    void compute_output();
    void compute_gradient();
    void add_parent(Layer* l);
};

class CrossEntropyError : public LossLayer {
public:
    /// The Cross Entropy Error loss layer, it is supervised
    CrossEntropyError() : LossLayer(true) {};
    void initialize(int batch_size, bool use_gpu, bool use_cpu);
    void compute_output();
    void compute_gradient();
    void add_parent(Layer* l);
};


class WeightInitializer {
public:
    enum Type {
        random_uni
        //random_normal
    };
    /// Weight initializer is necessary for each layer, as it initializes parameters, currently only random uniform initialization
    /// is permitted
    /// \param type
    WeightInitializer(Type type);
    void initialize(tensor_ptr weight_tensor);
private:
    std::random_device                      rd;
    std::mt19937                            mt;
    std::uniform_real_distribution<double>  dist_uniform;
    Type type;
};


#endif //PROJECT_LAYERS_H
