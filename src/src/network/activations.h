//
// Created by adam on 9.11.17.
//

#ifndef PROJECT_ACTIVATIONS_H
#define PROJECT_ACTIVATIONS_H

#include "tensor.h"
#include "tensor_gpu.h"
#include "activations_gpu.h"

using namespace std;

/**
 * Class representing the activation functions of neurons in the network
 */
class Activation {
public:
    enum Function {
        relu,
        leaky_relu,
        sigmoid,
        softmax
    };

    /// Default constructor, set activation type, also shape of the layer, and select placement
    /// \param type used function
    /// \param units shape of the layer in which this activation function is used
    /// \param use_gpu
    /// \param use_cpu
    Activation(Function type, t_shape* units, bool use_gpu, bool use_cpu); //batch size is the last dimension
    /// get number of trainable parameters (some activations can have parameters, like slope etc.)
    int get_n_params();
    /// get the output of the neurons
    tensor_ptr get_activity();
    /// get nth tensor of parameters
    tensor_ptr get_param(int n);
    /// used in forward pass
    void compute_activity(tensor_ptr activation);
    /// used in backwards pass
    void compute_gradient(tensor_ptr activation, tensor_ptr grad_error_wrt_output_prev);
    tensor_ptr get_gradient_output_wrt_activ();
    tensor_ptr get_gradient_error_wrt_activ();
    tensor_ptr get_gradient_error_wrt_param(int id);
    // CUDA FUNCTIONS
    void cu_initialize();
    cu_tensor cu_get_activity();
    cu_tensor cu_get_param(int n);
    void cu_compute_activity(cu_tensor activation);
    void cu_compute_gradient(cu_tensor activation, cu_tensor grad_error_wrt_output_prev);
    cu_tensor cu_get_gradient_output_wrt_activ();
    cu_tensor cu_get_gradient_error_wrt_activ();
    cu_tensor cu_get_gradient_error_wrt_param(int id);
private:
    bool use_gpu, use_cpu;
    Function type;
    int n_params;
    int batch_size;
    t_shape* shape;
    vector<tensor_ptr> params;
    tensor_ptr activity;
    tensor_ptr grad_output_wrt_activ;
    tensor_ptr grad_error_wrt_activ;
    vector<tensor_ptr> grad_error_wrt_param;

    vector<cu_tensor> cu_params;
    cu_tensor cu_activity;
    cu_tensor cu_grad_output_wrt_activ;
    cu_tensor cu_grad_error_wrt_activ;
    vector<cu_tensor> cu_grad_error_wrt_param;
};

tensor_ptr activation_relu(tensor_ptr input);
void apply_activation(tensor_ptr data, Activation act);
void get_gradient();

#endif //PROJECT_ACTIVATIONS_H
