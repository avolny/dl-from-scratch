//
// Created by adam on 9.11.17.
//

#include "activations.h"

float fun_relu(float in) {
    if(in < 0) {
        return 0;
    } else {
        return in;
    }
}

float fun_leaky_relu(float in) {
    if(in < 0) {
        return 0.1*in;
    } else {
        return in;
    }
}

float fun_sigmoid(float in) {
    return 1/(1 + std::exp(-in));
}

Activation::Activation(Activation::Function type, t_shape* units, bool use_gpu, bool use_cpu) :
        type(type), shape(units), use_gpu(use_gpu), use_cpu(use_cpu) {
    this->n_params = 0;
    this->batch_size = shape->dims[shape->n_dims-1];
    if(use_gpu)
        cu_initialize();
}


void Activation::cu_initialize() {
    cu_activity = cu_alloc_tensor(*shape);
    cu_grad_output_wrt_activ = cu_alloc_tensor(*shape);
    cu_grad_error_wrt_activ = cu_alloc_tensor(*shape);

    if(get_n_params() > 0)
        throw runtime_error("void Activation::cu_initialize() not implemented for params");
}


int Activation::get_n_params() {
    return this->n_params;
}

tensor_ptr Activation::get_activity() {
    return this->activity;
}

void Activation::compute_activity(tensor_ptr activation) {
    tensor_fun_unary f = nullptr;

    if(type == Function::softmax) {
        if(shape->n_dims == 2) {
            activity = activation->copy();
            float max = activity->max();
            activity->apply_fun(&max, [](float a, void* max) -> float {return exp(a-*(float*)max);}); //numerically stable softmax
            tensor2d_ptr activity2d = dynamic_pointer_cast<tensor2d>(activity);


            // compute sum for each batch and then normalize each vector of outputs by it
            for (int i = 0; i < batch_size; ++i) {
                float sum = 0;
                for (int j = 0; j < shape->dims[0]; ++j) {
                    sum += activity2d->array[j][i];
                }

                for (int j = 0; j < shape->dims[0]; ++j) {
                    activity2d->array[j][i] /= sum;
                }
            }

            return;
        } else {
            throw std::runtime_error("n/a");
        }
    } else if(this->type == Function ::relu) {
        f = fun_relu;
    } else if(type == Function::leaky_relu) {
        f = fun_leaky_relu;
    } else if(type == Function::sigmoid) {
        f = fun_sigmoid;
    }


    this->activity = activation->copy();
    this->activity->apply_fun(f);

}

void Activation::compute_gradient(tensor_ptr activation, tensor_ptr grad_error_wrt_output_prev) {
    tensor_fun_binary f = nullptr;

    if(this->type == Function::softmax) {
        if(shape->n_dims == 2) {
            tensor2d_ptr grad_error_wrt_activ = tensor2d_ptr(new tensor2d(*shape));
            tensor2d_ptr grad_error_wrt_output_prev2d = dynamic_pointer_cast<tensor2d>(grad_error_wrt_output_prev);
            tensor2d_ptr activity2d = dynamic_pointer_cast<tensor2d>(activity);
            int n_outs = shape->dims[0];

            tensor2d_ptr sm_derivatives = tensor2d_ptr(new tensor2d(n_outs, n_outs));
            for (int k = 0; k < batch_size; ++k) {

                for (int i = 0; i < n_outs; ++i) {
                    for (int j = 0; j < n_outs; ++j) {
                        if(i == j) { //dY_i/dXsi_i
                            float val = grad_error_wrt_output_prev2d->array[j][k]*activity2d->array[i][k]*(1-activity2d->array[i][k]);
                            grad_error_wrt_activ->array[i][k] += val;
                        } else { //dY_i/dXsi_j
                            float val = -grad_error_wrt_output_prev2d->array[j][k]*activity2d->array[i][k]*activity2d->array[j][k];
                            grad_error_wrt_activ->array[i][k] += val;
                        }
                    }
                }
            }

            this->grad_error_wrt_activ = grad_error_wrt_activ;
            
        } else {
            throw std::runtime_error("n/a");
        }


        return;
    } else if(this->type == Function::relu)
        f = [](float activation, float activity) -> float {if(activation > 0) return 1; else return 0;};
    else if(type == Function::leaky_relu)
        f = [](float activation, float activity) -> float {if(activation > 0) return 1; else return 0.1;};
    else if(type == Function::sigmoid)
        f = [](float activation, float activity) -> float {return activity*(1-activity);};


    this->grad_output_wrt_activ = activation->copy();
    this->grad_output_wrt_activ->apply_fun(this->activity, f);
    grad_error_wrt_activ = grad_output_wrt_activ->copy();
    grad_error_wrt_activ->multiply(grad_error_wrt_output_prev);
}

tensor_ptr Activation::get_gradient_output_wrt_activ() {
    return this->grad_output_wrt_activ;
}

tensor_ptr Activation::get_gradient_error_wrt_param(int id) {
    return nullptr;
}

tensor_ptr Activation::get_param(int n) {
    return params[n];
}

tensor_ptr Activation::get_gradient_error_wrt_activ() {
    return grad_error_wrt_activ;
}

cu_tensor Activation::cu_get_activity() {
    return cu_activity;
}

cu_tensor Activation::cu_get_param(int n) {
    return cu_params[n];
}

void Activation::cu_compute_activity(cu_tensor activation) {
    int fun;
    switch(type) {
        case relu: fun=FUNCTION_RELU; break;
        case leaky_relu: fun=FUNCTION_LEAKY_RELU; break;
        case sigmoid: fun=FUNCTION_SIGMOID; break;
        case softmax: fun=FUNCTION_SOFTMAX; break;
        default: fun=-1; break;
    }

    w_activations_compute_activity(activation, cu_activity, fun, NULL, 0);


}

void Activation::cu_compute_gradient(cu_tensor activation, cu_tensor grad_error_wrt_output_prev) {
    int fun;
    switch(type) {
        case relu: fun=FUNCTION_RELU; break;
        case leaky_relu: fun=FUNCTION_LEAKY_RELU; break;
        case sigmoid: fun=FUNCTION_SIGMOID; break;
        case softmax: fun=FUNCTION_SOFTMAX; break;
        default: fun=-1; break;
    }

    w_activations_compute_grad_wrt_activity(activation, cu_activity, grad_error_wrt_output_prev,
                                            cu_grad_output_wrt_activ, cu_grad_error_wrt_activ, fun, NULL, 0);
}

cu_tensor Activation::cu_get_gradient_output_wrt_activ() {
    return cu_grad_output_wrt_activ;
}

cu_tensor Activation::cu_get_gradient_error_wrt_activ() {
    return cu_grad_error_wrt_activ;
}

cu_tensor Activation::cu_get_gradient_error_wrt_param(int id) {
    throw runtime_error("cu_tensor Activation::cu_get_gradient_error_wrt_param NOT IMPLEMENTED");
    return cu_tensor();
}
