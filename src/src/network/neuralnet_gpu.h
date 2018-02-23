#ifndef _NETWORK_NEURALNET_GPU_
#define _NETWORK_NEURALNET_GPU_


#include "tensor_gpu.h"


void w_sgd_update_params(cu_tensor params, cu_tensor grad, float learning_rate, float grad_clip);
void w_rmsprop_update_params(cu_tensor params, cu_tensor grad, cu_tensor mean_square, float learning_rate, float decay, float eps, float grad_clip);


#endif //_NETWORK_NEURALNET_GPU_