//
// Created by adam on 8.11.17.
//

#include "layers.h"
//#include "gpu.h"

Layer::Layer(t_shape input_shape, t_shape output_shape, string name) : name(name) {
    children = vector<Layer*>();
    parents = vector<Layer*>();
    this->input_shape = new t_shape(input_shape);
    this->output_shape = new t_shape(output_shape);
}

Layer::Layer(string name) : Layer(t_shape(), t_shape(), "") {

}

Layer::Layer() : Layer("") {

}

int Layer::get_num_trainable() {
    return 0;
}

t_shape Layer::get_trainable_shape(int n) {
    return t_shape();
}

void Layer::set_input_shape(t_shape input_shape) {
    this->input_shape = new t_shape(input_shape);
}

void Layer::set_output_shape(t_shape output_shape) {
    this->output_shape = new t_shape(output_shape);
}

void Layer::initialize(int batch_size, bool use_gpu, bool use_cpu) {
    if(initialized)
        return;

    this->batch_size = batch_size;

    for (int i = 0; i < parents.size(); ++i) {
        parents[i]->initialize(batch_size, use_gpu, use_cpu);
    }

    this->use_gpu = use_gpu;
    this->use_cpu = use_cpu;
    initialized = true;

}

void Layer::reset_output_computation() {
    if(computed_output) {
        computed_output = false;
        for (int i = 0; i < parents.size(); ++i) {
            parents[i]->reset_output_computation();
        }

    }
}

void Layer::compute_output() {
    if(use_gpu)
        cudaDeviceSynchronize();
//    cout << "layer " << name << " output computed " << endl;
    computed_output = true;
}

void Layer::compute_gradient() {
    if(use_gpu)
        cudaDeviceSynchronize();
//    cout << "layer " << name << " gradient computed " << endl;
    computed_grad = true;
}

void Layer::add_parent(Layer *l) {
    parents.push_back(l);
}

void Layer::add_child(Layer *l) {
    children.push_back(l);
}

tensor_ptr Layer::get_trainable(int n) {
    return nullptr;
}

void Layer::reset_grad_computation() {
    if(computed_grad) {
        computed_grad = false;
        for (int i = 0; i < parents.size(); ++i) {
            parents[i]->reset_grad_computation();
        }
    }
}

tensor_ptr Layer::get_grad_output_wrt_activ() {
    return tensor_ptr();
}

tensor_ptr Layer::get_grad_error_wrt_output() {
    return tensor_ptr();
}

tensor_ptr Layer::get_grad_error_wrt_param(int n) {
    return tensor_ptr();
}

cu_tensor Layer::cu_get_grad_output_wrt_activ() {
    return cu_tensor();
}

cu_tensor Layer::cu_get_grad_error_wrt_output() {
    return cu_tensor();
}


cu_tensor Layer::cu_get_grad_error_wrt_param(int n) {
    return cu_tensor();
}

cu_tensor Layer::cu_get_trainable(int n) {
    return cu_tensor();
}

cu_shape Layer::cu_get_trainable_shape(int n) {
    return cu_shape();
}

Layer::~Layer() {
    delete input_shape;
    delete output_shape;
}


int InputLayer::static_id = 0;

InputLayer::InputLayer(t_shape* input_shape) : Layer() {
    stringstream ss;
    ss << "Input_" << static_id++;
    name = ss.str();
    data_shape = input_shape;
}

tensor_ptr InputLayer::get_output() {
    return batch;
}

bool InputLayer::is_input() {
    return true;
}

int InputLayer::get_num_trainable() {
    return 0;
}

void InputLayer::set_input(tensor_ptr input) {
    if(!output_shape->eq(input->shape)) {
        input->shape.print();
        output_shape->print();
        throw std::invalid_argument( "mismatching data dimensions" );
    }

    batch = input;
}

void InputLayer::initialize(int batch_size, bool use_gpu, bool use_cpu) {
    Layer::initialize(batch_size, use_gpu, use_cpu);

    int n_dims = data_shape->n_dims+1;
    int* dims = new int[n_dims];
    for (int i = 0; i < n_dims-1; ++i) {
        dims[i] = data_shape->dims[i];
    }

    dims[n_dims-1] = batch_size;
    output_shape = new t_shape(dims, n_dims);
    input_shape = new t_shape();

    if(use_gpu) {
        cu_batch.array = nullptr;
        // initialization of the input array has to be done before passing it to the InputLayer
//        CHECK_ERROR( cudaMalloc((void**) &cu_batch, sizeof(float)*output_shape->flat_size()) );
    }
}

void InputLayer::reset_output_computation() {
    if(use_gpu) {
        CHECK_ERROR(cudaFree(cu_batch.array));
    }
}

void InputLayer::compute_output() {
    Layer::compute_output();
}

void InputLayer::add_parent(Layer *l) {
    Layer::add_parent(l);
}

void InputLayer::compute_gradient() {
    for (int i = 0; i < children.size(); ++i) {
        if(!children[i]->computed_grad) {
            children[i]->compute_gradient();
        }
    }
    
    Layer::compute_gradient();
}

cu_tensor InputLayer::cu_get_output() {
    return cu_batch;
}

void InputLayer::cu_set_input(cu_tensor input) {
    cu_batch = input;
}

InputLayer::~InputLayer() {
    if (cu_batch.array) CHECK_ERROR(cudaFree(cu_batch.array));
}


int DenseLayer::static_id = 0;

DenseLayer::DenseLayer(int n_outs, Activation::Function act, WeightInitializer* winit) : Layer() {
    stringstream ss;
    ss << "Dense_" << static_id++;
    name = ss.str();
    this->n_outs = n_outs;
    this->activation_function = act;
    this->winit = winit;
}

tensor_ptr DenseLayer::get_output() {
    return activation->get_activity();
}

bool DenseLayer::is_input() {
    return false;
}

int DenseLayer::get_num_trainable() {
    return 2;
}

void DenseLayer::compute_output() {
    if(!parents[0]->computed_output)
        parents[0]->compute_output();

    if(use_cpu) {
        tensor_ptr in = (tensor_ptr) parents[0]->get_output();
        activation_tensor = weight_tensor->dot(in);
        activation_tensor->add(bias_tensor->repeat(1, batch_size));
        activation->compute_activity(activation_tensor);
    }
    if(use_gpu) {
        cu_tensor cu_input = parents[0]->cu_get_output();
        w_dense_compute_activation(cu_weight_tensor, cu_bias_tensor, cu_input, cu_activation_tensor);
        activation->cu_compute_activity(cu_activation_tensor);
    }

    Layer::compute_output();
}


void DenseLayer::compute_gradient() {
    if(!children[0]->computed_grad)
        children[0]->compute_gradient();

    if(use_cpu) {
        activation->compute_gradient(activation_tensor, children[0]->get_grad_error_wrt_output());
        grad_error_wrt_weights = activation->get_gradient_error_wrt_activ()->dot(
                parents[0]->get_output()->transpose()); // weights grad
        grad_error_wrt_bias = activation->get_gradient_error_wrt_activ()->sum(1);
        grad_error_wrt_output = weight_tensor->transpose()->dot(activation->get_gradient_error_wrt_activ()); // delta_i
    }
    if(use_gpu){
        activation->cu_compute_gradient(cu_activation_tensor, children[0]->cu_get_grad_error_wrt_output());
        cudaDeviceSynchronize();
        cu_tensor input = parents[0]->cu_get_output();
        cu_tensor grad_err_wrt_activ = activation->cu_get_gradient_error_wrt_activ();

        w_dense_compute_grad_error_wrt_weights(grad_err_wrt_activ, input, cu_grad_error_wrt_weights);
        cudaDeviceSynchronize();
        w_dense_compute_grad_error_wrt_bias(grad_err_wrt_activ, cu_grad_error_wrt_bias);
        cudaDeviceSynchronize();
        w_dense_compute_grad_error_wrt_output(cu_weight_tensor, grad_err_wrt_activ, cu_grad_error_wrt_output);
        cudaDeviceSynchronize();

    }

    Layer::compute_gradient();
}

void DenseLayer::initialize(int batch_size, bool use_gpu, bool use_cpu) {
    Layer::initialize(batch_size, use_gpu, use_cpu);

    input_shape = parents[0]->output_shape;
    output_shape = new t_shape({n_outs, batch_size});

    weight_tensor = tensor2d_ptr(new tensor2d(output_shape->dims[0], input_shape->dims[0]));
    bias_tensor = tensor2d_ptr(new tensor2d(output_shape->dims[0], 1));
    winit->initialize(weight_tensor);
    bias_tensor->fill(0.);
    activation = new Activation(activation_function, output_shape, use_gpu, use_cpu);
    activation_tensor = nullptr;

//    cout << "dense init" << endl;

    if(use_gpu) {
        cu_weight_tensor = tensor_to_cu(weight_tensor);
        cu_bias_tensor = tensor_to_cu(bias_tensor);
        cu_activation_tensor = cu_alloc_tensor(*output_shape);

        cu_grad_error_wrt_weights = cu_alloc_tensor(cu_weight_tensor.shape);
        cu_grad_error_wrt_bias = cu_alloc_tensor(cu_bias_tensor.shape);
        cu_grad_error_wrt_output = cu_alloc_tensor(*input_shape);
    }

//    this->activation_tensor = new tensor2d({this->n_outs, this->batch_size});
}


void DenseLayer::add_parent(Layer *l) {
    if(parents.size() == 1) {
        throw std::invalid_argument("dense layer cannot have more than 1 parent");
    }
    Layer::add_parent(l);
}

t_shape DenseLayer::get_trainable_shape(int n) {
    if(n == 0) {
        return weight_tensor->shape;
    } else if(n == 1) {
        return bias_tensor->shape;
    } else {
        return activation->get_param(n-2)->shape;
    }

//    return Layer::get_trainable_shape(n-2);
}

cu_shape DenseLayer::cu_get_trainable_shape(int n) {
    if(n == 0) {
        return cu_weight_tensor.shape;
    } else if(n == 1) {
        return cu_bias_tensor.shape;
    } else {
        return activation->cu_get_param(n-2).shape;
    }
}

tensor_ptr DenseLayer::get_grad_output_wrt_activ() {
    return activation->get_gradient_output_wrt_activ();
}

tensor_ptr DenseLayer::get_grad_error_wrt_output() {
    return grad_error_wrt_output;
}

tensor_ptr DenseLayer::get_grad_error_wrt_param(int n) {
    if(n == 0) {
        return grad_error_wrt_weights;
    } else if(n == 1) {
        return grad_error_wrt_bias;
    } else {
        return activation->get_gradient_error_wrt_param(n-2);
    }

}

tensor_ptr DenseLayer::get_trainable(int n) {
    if(n == 0) {
        return weight_tensor;
    } else if(n == 1) {
        return bias_tensor;
    } else {
        return activation->get_param(n-2);
    }
}

cu_tensor DenseLayer::cu_get_trainable(int n) {
    if(n == 0) {
        return cu_weight_tensor;
    } else if(n == 1) {
        return cu_bias_tensor;
    } else {
        return activation->cu_get_param(n-2);
    }
}

cu_tensor DenseLayer::cu_get_output() {
    return activation->cu_get_activity();
}

cu_tensor DenseLayer::cu_get_grad_output_wrt_activ() {
    return activation->cu_get_gradient_output_wrt_activ();
}

cu_tensor DenseLayer::cu_get_grad_error_wrt_output() {
    return cu_grad_error_wrt_output;
}

cu_tensor DenseLayer::cu_get_grad_error_wrt_param(int n) {
    if(n == 0) {
        return cu_grad_error_wrt_weights;
    } else if(n == 1) {
        return cu_grad_error_wrt_bias;
    } else {
        return activation->cu_get_gradient_error_wrt_param(n-2);
    }
}

DenseLayer::~DenseLayer() {
    if(cu_activation_tensor.array) CHECK_ERROR(cudaFree(cu_activation_tensor.array));
    if(cu_weight_tensor.array) CHECK_ERROR(cudaFree(cu_weight_tensor.array));
    if(cu_bias_tensor.array) CHECK_ERROR(cudaFree(cu_bias_tensor.array));
    if(cu_grad_error_wrt_output.array) CHECK_ERROR(cudaFree(cu_grad_error_wrt_output.array));
    if(cu_grad_error_wrt_weights.array) CHECK_ERROR(cudaFree(cu_grad_error_wrt_weights.array));
    if(cu_grad_error_wrt_bias.array) CHECK_ERROR(cudaFree(cu_grad_error_wrt_bias.array));
    delete activation;
    delete winit;
}

WeightInitializer::WeightInitializer(WeightInitializer::Type type) : rd{}, mt{rd()}, dist_uniform{-0.1,0.1} {
    this->type = type;
}

void WeightInitializer::initialize(tensor_ptr weight_tensor) {

    if(weight_tensor->shape.n_dims == 2) {

        tensor2d_ptr wt = std::dynamic_pointer_cast<tensor2d>(weight_tensor);

        if(type == WeightInitializer::Type::random_uni) {
            for (int i = 0; i < wt->rows; ++i) {
                for (int j = 0; j < wt->cols; ++j) {
                    wt->array[i][j] = dist_uniform(mt);
                }
            }
        }
    } else if (weight_tensor->shape.n_dims == 1) {
        tensor1d_ptr wt = std::dynamic_pointer_cast<tensor1d>(weight_tensor);

        if(type == WeightInitializer::Type::random_uni) {
            for (int i = 0; i < wt->rows; ++i) {
                wt->array[i] = dist_uniform(mt);
            }
        }
    }
}


void LossLayer::set_target(tensor_ptr target) {
    if(!target->shape.eq(input_shape)) {
        target->shape.print();
        input_shape->print();
        throw std::runtime_error("void LossLayer::set_target(tensor_ptr target) loss layer target tensor set dimension mismatch");
    }
    this->target = target;
}

void LossLayer::cu_set_target(cu_tensor target) {
    if(!input_shape->eq(target.shape)) {
        cu_shape_print(target.shape);
        input_shape->print();
        throw std::runtime_error("void LossLayer::cu_set_target(cu_tensor target) loss layer target tensor set dimension mismatch");
    }

    this->cu_target = target;
}

cu_tensor LossLayer::cu_get_output() {
    return cu_loss_sum;
}

cu_tensor LossLayer::cu_get_grad_error_wrt_output() {
    return cu_loss_grad;
}

void LossLayer::initialize(int batch_size, bool use_gpu, bool use_cpu) {
    Layer::initialize(batch_size, use_gpu, use_cpu);
    cu_target.array = nullptr;
}

void LossLayer::reset_output_computation() {
    if(cu_target.array)
        CHECK_ERROR(cudaFree(cu_target.array));
    Layer::reset_output_computation();
}

LossLayer::~LossLayer() {
    if(cu_target.array) CHECK_ERROR(cudaFree(cu_target.array));
    if(cu_loss.array) CHECK_ERROR(cudaFree(cu_loss.array));
    if(cu_loss_grad.array) CHECK_ERROR(cudaFree(cu_loss_grad.array));
    if(cu_loss_sum.array) CHECK_ERROR(cudaFree(cu_loss_sum.array));
}

void MSE::initialize(int batch_size, bool use_gpu, bool use_cpu) {
    LossLayer::initialize(batch_size, use_gpu, use_cpu);

    input_shape = parents[0]->output_shape;
    output_shape = new t_shape({1});

    if(use_gpu) {
        cu_target = cu_alloc_tensor(*input_shape); //tensor for correct output
        cu_loss = cu_alloc_tensor(*input_shape);
        cu_loss_grad = cu_alloc_tensor(*input_shape);
        cu_loss_sum = cu_alloc_tensor(new_cu_shape(1)); // this holds the single output value
    }
}

void MSE::compute_output() {
    if(!parents[0]->computed_output)
        parents[0]->compute_output();

    if(use_cpu) {
        tensor_ptr in = parents[0]->get_output();
        tensor_ptr loss = in->copy();
        loss->sub(target);
        loss->multiply(loss);
        this->loss = new_tensor(t_shape({1}));
        this->loss->fill(loss->mean() * 0.5f);
    }
    if(use_gpu) {
        cu_tensor in = parents[0]->cu_get_output();
        w_mse_compute_activation(in, cu_target, cu_loss, cu_loss_sum);
    }

    Layer::compute_output();
}

void MSE::compute_gradient() {
//    cout << "MSE gradient" << endl;
    if(use_cpu) {
        loss_grad = parents[0]->get_output()->copy();
        loss_grad->sub(target);
        loss_grad->multiply_const(1.0f / input_shape->flat_size());
    }
    if(use_gpu) {
        cu_tensor in = parents[0]->cu_get_output();
        w_mse_compute_grad_error_wrt_output(in, cu_target, cu_loss_grad);
    }

//    loss_grad->print();

    Layer::compute_gradient();
}

void MSE::add_parent(Layer *l) {
    if(parents.size() == 1) {
        throw std::runtime_error("MSE layer can have only a single parent");
    }
    Layer::add_parent(l);
}


void CrossEntropyError::initialize(int batch_size, bool use_gpu, bool use_cpu) {
    LossLayer::initialize(batch_size, use_gpu, use_cpu);

    input_shape = parents[0]->output_shape;
    output_shape = new t_shape({1});

    if(use_gpu) {
        cu_target = cu_alloc_tensor(*input_shape); //tensor for correct output
        cu_loss = cu_alloc_tensor(*input_shape);
        cu_loss_grad = cu_alloc_tensor(*input_shape);
        cu_loss_sum = cu_alloc_tensor(new_cu_shape(1)); // this holds the single output value
    }
}

void CrossEntropyError::compute_output() {
    if(!parents[0]->computed_output)
        parents[0]->compute_output();

    if(use_cpu) {
        tensor_ptr in = parents[0]->get_output();
        tensor_ptr loss = in->copy();
        loss->multiply(target);
        loss->apply_fun([](float a) -> float { return (a != 0) ? -std::log(a + 1e-10) : 0; });
        this->loss = new_tensor(t_shape({1}));
        this->loss->fill(loss->mean());
    }
    if(use_gpu) {
        cu_tensor in = parents[0]->cu_get_output();
        w_crossentropy_compute_activation(in, cu_target, cu_loss, cu_loss_sum);
    }
    Layer::compute_output();
}

void CrossEntropyError::compute_gradient() {
    if(use_cpu) {
        loss_grad = parents[0]->get_output()->copy();
        loss_grad->multiply(target);
        loss_grad->apply_fun([](float a) -> float { return (a > 0) ? -1. / (a + 1e-10) : 0; }); //avoid zeros
        loss_grad->multiply_const(1.0f / input_shape->flat_size()); // mean
//        loss_grad->print("loss_grad cpu");
    }
    if(use_gpu){
//        cout << "crossentropy compute gradient " << endl;
        cu_tensor in = parents[0]->cu_get_output();
        w_crossentropy_compute_grad_error_wrt_output(in, cu_target, cu_loss_grad);
        cudaDeviceSynchronize();
//        tensor_from_cu(cu_loss_grad)->print("cu_loss_grad gpu");
//        cout << "crossentropy compute gradient done" << endl;
    }
    Layer::compute_gradient();
}

void CrossEntropyError::add_parent(Layer *l) {
    if(parents.size() == 1) {
        throw std::runtime_error("MSE layer can have only a single parent");
    }
    Layer::add_parent(l);
}
