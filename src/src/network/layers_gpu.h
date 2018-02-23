#ifndef _LAYERS_GPU_
#define _LAYERS_GPU_

#include "tensor_gpu.h"

//w stands for wrapper
void w_crossentropy_compute_activation(cu_tensor input, cu_tensor target, cu_tensor loss, cu_tensor loss_sum);
void w_crossentropy_compute_grad_error_wrt_output(cu_tensor input, cu_tensor target, cu_tensor loss_grad);

void w_mse_compute_activation(cu_tensor input, cu_tensor target, cu_tensor loss, cu_tensor loss_sum);
void w_mse_compute_grad_error_wrt_output(cu_tensor input, cu_tensor target, cu_tensor loss_grad);


void w_dense_compute_activation(cu_tensor weights, cu_tensor bias, cu_tensor input, cu_tensor activation); // activation is output
void w_dense_compute_grad_error_wrt_weights(cu_tensor grad_err_wrt_activ, cu_tensor input, cu_tensor grad_err_wrt_weights);
void w_dense_compute_grad_error_wrt_bias(cu_tensor grad_err_wrt_activ, cu_tensor grad_err_wrt_bias);
void w_dense_compute_grad_error_wrt_output(cu_tensor weights, cu_tensor grad_err_wrt_activ, cu_tensor grad_err_wrt_output);



#endif //_LAYERS_GPU_