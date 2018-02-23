#ifndef _NETWORK_ACTIVATIONS_GPU
#define _NETWORK_ACTIVATIONS_GPU

#include "tensor_gpu.h"


#define FUNCTION_RELU 0
#define FUNCTION_SIGMOID 1
#define FUNCTION_LEAKY_RELU 2
#define FUNCTION_SOFTMAX 3


void w_activations_compute_activity(cu_tensor activation, cu_tensor activity, int function, cu_tensor* params, int n_params);
void w_activations_compute_grad_wrt_activity(cu_tensor activation, cu_tensor activity, cu_tensor grad_err_wrt_output_prev,
         cu_tensor grad_output_wrt_activity, cu_tensor grad_err_wrt_activity, int function, cu_tensor* params, int n_params);


#endif //_NETWORK_ACTIVATIONS_GPU