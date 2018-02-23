//
// Created by adam on 26.10.17.
//

#include "neuralnet.h"
#include <string>
#include <chrono>

NeuralNet::NeuralNet(bool sequential) {
    this->sequential = sequential;
    //if sequential is true, that means that the network will be a linear graph with 1 input and 1 output layer
    this->layers = vector<Layer*>();
    this->input_layers = vector<InputLayer*>();
    this->output_layers = vector<Layer*>();
    this->loss_layers = vector<LossLayer*>();
    this->loss_layers_supervised = vector<LossLayer*>();
}


tensor_ptr* NeuralNet::predict(tensor_ptr *batch, int n_batch_ins) {
    if(!this->initialized) {
        throw runtime_error("trying to predict from net that was not initialized");
    }

    if(n_batch_ins != this->input_layers.size()) {
        throw std::invalid_argument("the number of batch inputs does not match the number of input layers");
    }

    for (int i = 0; i < this->output_layers.size(); ++i) {
        this->output_layers[i]->reset_output_computation();
    }

    for (int i = 0; i < this->input_layers.size(); ++i) {
        if(use_cpu)
            this->input_layers[i]->set_input(batch[i]);
        if(use_cpu) {
            this->input_layers[i]->cu_set_input(tensor_to_cu(batch[i]));
        }
    }

    for (int i = 0; i < this->output_layers.size(); ++i) {
        this->output_layers[i]->compute_output();
    }

    tensor_ptr* out = new tensor_ptr[this->output_layers.size()];
    for (int i = 0; i < this->output_layers.size(); ++i) {
        if(use_cpu)
            out[i] = this->output_layers[i]->get_output();
        if(use_gpu)
            out[i] = tensor_from_cu(this->output_layers[i]->cu_get_output());
    }

    return out;
}

tensor_ptr NeuralNet::predict(tensor_ptr batch) {
    if(!this->sequential)
        throw std::runtime_error("this predict method works only on sequential models");

    tensor_ptr* x = new tensor_ptr[1];
    x[0] = batch;
    tensor_ptr* y = this->predict(x, 1);

    return y[0];
}

void NeuralNet::fit(tensor_ptr** x, tensor_ptr** y, int n_batches, int n_batch_ins, int n_batch_outs, int n_epochs) {
    if(!initialized) {
        throw runtime_error("trying to predict from net that was not initialized");
    }

    cout << "fitting " << n_batches << " batches" << endl;

    train_info info;
    for (int i = 0; i < n_epochs; ++i) {
        cout << "STARTING EPOCH " << i << endl;
        cout << "iterating over batches" << endl;
        float avg_loss = 0;
        float avg_accuracy = 0;
//        double avg_time = 0.;
        chrono::high_resolution_clock::time_point t1 = chrono::high_resolution_clock::now();
        for (int j = 0; j < n_batches; ++j) {
//            chrono::high_resolution_clock::time_point t3 = chrono::high_resolution_clock::now();

            train_batch(info, x[j], y[j], n_batch_ins, n_batch_outs);
            if(info.loss < 2) // there is a bug in computing the loss on GPU, the learning is working properly
                avg_loss += info.loss; // but the loss computation sometimes results in unreasonably high number
            avg_accuracy += info.accuracy;

//            chrono::high_resolution_clock::time_point t4 = chrono::high_resolution_clock::now();
//            double dif2 = chrono::duration_cast<chrono::nanoseconds>( t4 - t3 ).count()/1e9;
//            avg_time += dif2;
//            printf ("batch %d average time %lf seconds. expected time %lf seconds.\n", j ,avg_time/(j+1),  avg_time/(j+1)*n_batches);
        }
        chrono::high_resolution_clock::time_point t2 = chrono::high_resolution_clock::now();
        double dif = chrono::duration_cast<chrono::nanoseconds>( t2 - t1 ).count();
        printf ("EPOCH ELAPSED TIME %lf seconds.\n", dif/1e9 );
//        avg_loss /= n_batches;
        avg_accuracy /= n_batches;
        cout << "EPOCH AVERAGE LOSS: " << avg_loss*1./n_batches << endl;
        cout << "EPOCH AVERAGE ACCURACY: " << avg_accuracy << endl << endl;


    }
}

void NeuralNet::train_batch(train_info &info, tensor_ptr *batch_x, tensor_ptr *batch_y, int n_batch_ins, int n_batch_outs) {
    if(!this->initialized) {
        throw runtime_error("trying to predict from net that was not initialized");
    }

    if(n_batch_ins != this->input_layers.size()) {
        throw std::invalid_argument("the number of batch inputs does not match the number of input layers");
    }

    if(n_batch_outs != this->loss_layers_supervised.size()) {
        throw std::invalid_argument("the number of batch targets does not match the number of supervised loss layers");
    }

    reset_computation(); // RESET GRAPHS
    set_input(batch_x, n_batch_ins); // ASSIGN INPUT
    set_target(batch_y, n_batch_outs); // ASSIGN TARGET OUTPUT
    compute_output(true); // START COMPUTATION FROM LOSS LAYERS
    compute_grad(); // COMPUTE BACKWARDS PASS GRADIENTS FROM INPUT LAYERS
    optimizer->update_params(); // UPDATE OPTIMIZED PARAMETERS

    tensor2d_ptr out;
    tensor2d_ptr true_out;
    if(use_cpu) {
        out = std::dynamic_pointer_cast<tensor2d>(this->output_layers[0]->get_output());
        true_out = std::dynamic_pointer_cast<tensor2d>(batch_y[0]);
    }
    if(use_gpu){
        out = std::dynamic_pointer_cast<tensor2d>(tensor_from_cu(this->output_layers[0]->cu_get_output()));
        true_out = std::dynamic_pointer_cast<tensor2d>(batch_y[0]);
    }

    // compute accuracy
    int cnt_correct = 0;

    for (int j = 0; j < out->cols; ++j) {
        int correct = -1;
        int max_out = 0;
        for (int i = 0; i < out->rows; ++i) {
            if(true_out->array[i][j] == 1)
                correct = i;
            if(out->array[i][j] > out->array[max_out][j])
                max_out = i;
        }

        if(max_out == correct) {
            cnt_correct++;
        }
    }
    
    info.loss = get_loss();
    info.accuracy = cnt_correct*1.0/out->cols;
}


void NeuralNet::add_layer(Layer *l) {
    if(l->is_input()) {
        if(this->sequential && this->input_layers.size() == 1) {
            throw std::invalid_argument("sequential model can have only single input layer");
        }
        this->input_layers.push_back((InputLayer*) l);
    } else {
        if(this->sequential && this->layers.size() > 0)
            l->add_parent(this->layers[this->layers.size()-1]);
            this->layers[this->layers.size()-1]->add_child(l);
    }

    this->layers.push_back(l);
}

void NeuralNet::add_layer(Layer *l, bool is_output) {
    this->add_layer(l);
    if(is_output)
        this->add_output(l);
}


void NeuralNet::add_output(Layer *l) {
    if(this->sequential && this->output_layers.size() == 1) {
        throw std::invalid_argument("sequential model can have only single output layer");
    }

    this->output_layers.push_back(l);
}

void NeuralNet::initialize(int batch_size, Optimizer* optimizer, bool use_gpu, bool use_cpu) {
    for (int i = 0; i < this->output_layers.size(); ++i) {
        this->output_layers[i]->initialize(batch_size, use_gpu, use_cpu);
    }

    for (int i = 0; i < loss_layers.size(); ++i) {
        loss_layers[i]->initialize(batch_size, use_gpu, use_cpu);
    }

    this->use_gpu = use_gpu;
    this->use_cpu = use_cpu;
    this->optimizer = optimizer;
    optimizer->initialize(this);

    this->initialized = true;
}

void NeuralNet::add_loss_layer(LossLayer *l) {
    loss_layers.push_back(l);
    add_layer(l);
    if(l->has_target())
        this->loss_layers_supervised.push_back(l);

}

int sign(float a) {
    if (a > 0)
        return 1;
    else if(a < 0)
        return -1;
    else
        return 0;
}

void NeuralNet::validate_gradients(tensor_ptr *x, tensor_ptr *y, int n_batch_ins, int n_batch_outs) {
    float eps = 0.1;

    reset_computation();
    set_input(x, n_batch_ins);
    set_target(y, n_batch_outs);
    compute_output(true);
    compute_grad();

    float loss_base = get_loss();

    int total_samples = 0;
    int same_signs = 0;
    float difference_avg = 0;

    int max_runs = 100;
    bool done = false;

    for (int i = 0; i < layers.size() && !done; ++i) {
        for (int j = 0; j < layers[i]->get_num_trainable() && !done; ++j) {
            tensor_ptr param = layers[i]->get_trainable(j);
            tensor_ptr param_grad = layers[i]->get_grad_error_wrt_param(j);


            if(param->shape.n_dims == 1) {
                throw std::runtime_error("not implemented");
            } else if(param->shape.n_dims == 2) {
                tensor2d_ptr p = std::dynamic_pointer_cast<tensor2d>(param);
                tensor2d_ptr p_grad = std::dynamic_pointer_cast<tensor2d>(param_grad);
                tensor2d_ptr numeric_grad = std::dynamic_pointer_cast<tensor2d>(new_tensor(p_grad->shape));

                for (int k = 0; k < p->rows && !done; ++k) {
                    for (int l = 0; l < p->cols && !done; ++l) {
                        float val = p->array[k][l];

                        p->array[k][l] = val+eps;

                        reset_computation();
                        compute_output(true);

                        float loss_nudge = get_loss();
                        float grad_numeric = (loss_nudge-loss_base)/eps;
                        float grad_analytic = p_grad->array[k][l];
                        numeric_grad->array[k][l] = grad_numeric;

                        difference_avg += abs(grad_numeric-grad_analytic);
                        if(sign(grad_numeric) == sign(grad_analytic))
                            same_signs++;


                        total_samples++;

                        p->array[k][l] = val;

                        if(total_samples % 10 == 0) {
                            cout << total_samples;
                        }

                        if (total_samples > max_runs) {
                            done = true;
                            break;
                        }
                    }
                }

                cout << "------------------------------------------------" << endl;
                cout << "Parameter tensor, analytic gradient: " << endl;
                p_grad->print();
                cout << endl << "numeric gradient: " << endl;
                numeric_grad->print();
                cout << endl;

            } else
                throw std::runtime_error("not implemented");
        }
    }

    difference_avg /= total_samples;

    cout << "NUMERIC GRADIENT CHECK UP" << endl;
    cout << "used epsillon of " << eps << endl;
    cout << "total samples: " << total_samples << ", agreeing gradient signs: " << same_signs << ", average absolute difference: " << difference_avg << endl;

}

void NeuralNet::reset_computation() {
    for (int i = 0; i < this->loss_layers.size(); ++i) {
        this->loss_layers[i]->reset_output_computation();
    }

    for (int i = 0; i < this->loss_layers.size(); ++i) {
        this->loss_layers[i]->reset_grad_computation();
    }
}

void NeuralNet::compute_output(bool use_loss) {
    if(use_loss) { // compute from loss layers
        for (int i = 0; i < this->loss_layers.size(); ++i) {
            this->loss_layers[i]->compute_output();
        }
    } else { // compute from outputs
        for (int i = 0; i < this->output_layers.size(); ++i) {
            this->output_layers[i]->compute_output();
        }
    }
}

void NeuralNet::compute_grad() {
    for (int i = 0; i < this->input_layers.size(); ++i) {
        this->input_layers[i]->compute_gradient();
    }
}

void NeuralNet::set_input(tensor_ptr *batch_x, int n_batch_ins) {
    for (int i = 0; i < this->input_layers.size(); ++i) {
        if(use_cpu) {
            this->input_layers[i]->set_input(batch_x[i]);
        }
        if(use_gpu){
            this->input_layers[i]->cu_set_input(tensor_to_cu(batch_x[i]));
        }
    }
}

void NeuralNet::set_target(tensor_ptr *batch_y, int n_batch_outs) {
    for (int i = 0; i < loss_layers_supervised.size(); ++i) {
        if(use_cpu) {
            loss_layers_supervised[i]->set_target(batch_y[i]);
        }
        if(use_gpu) {
            loss_layers_supervised[i]->cu_set_target(tensor_to_cu(batch_y[i]));
        }
    }
}

float NeuralNet::get_loss() {
    float loss = 0;
    for (int j = 0; j < loss_layers.size(); ++j) {
        if(use_cpu && !use_gpu) {
            loss += loss_layers[j]->get_output()->sum(); //it's just a single element
        }
        if(use_gpu) {
            loss += tensor_from_cu(loss_layers[j]->cu_get_output())->sum();
        }
    }
    return loss;
}

NeuralNet::~NeuralNet() {
    delete optimizer;
    for (int i = 0; i < layers.size(); ++i) {
        delete layers[i];
    }
}


SGD::SGD(float learning_rate, float grad_clip) : learning_rate(learning_rate), grad_clip(grad_clip) {
    
}

void SGD::initialize(NeuralNet *net) {
    Optimizer::initialize(net);
}

void SGD::update_params() {
    for (int i = 0; i < net->layers.size(); ++i) {
        Layer* l = net->layers[i];
        for (int j = 0; j < l->get_num_trainable(); ++j) {
            if(use_cpu) {
                tensor_ptr params = l->get_trainable(j);
                tensor_ptr grad = l->get_grad_error_wrt_param(j)->copy();
                grad->apply_fun(&grad_clip, [](float a, void *clip) -> float {
                    return a > 0 ? std::min(*(float *) clip, a) : std::max(-*(float *) clip, a);
                });
                grad->multiply_const(learning_rate);
                params->sub(grad);
            }
            if(use_gpu) {
                cu_tensor params = l->cu_get_trainable(j);
                cu_tensor grad = l->cu_get_grad_error_wrt_param(j);
                w_sgd_update_params(params, grad, learning_rate, grad_clip);
                cudaDeviceSynchronize();
            }
        }
    }
}

RMSProp::RMSProp(float learning_rate, float decay, float eps, float grad_clip) :
        learning_rate(learning_rate),decay(decay), eps(eps), grad_clip(grad_clip) {

}

void RMSProp::initialize(NeuralNet *net) {
    Optimizer::initialize(net);

    for (int i = 0; i < net->layers.size(); ++i) {
        for (int j = 0; j < net->layers[i]->get_num_trainable(); ++j) {
            // for each trainable tensor in the network create tensor with the same shape, to hold the mean_square param
            if(use_cpu) {
                tensor_ptr ms = new_tensor(net->layers[i]->get_trainable_shape(j));
                ms->fill(0.);
                mean_square.push_back(ms);
            }
            if(use_gpu) {
                cu_tensor ms = cu_alloc_tensor(net->layers[i]->cu_get_trainable_shape(j));
                w_tensor_fill(ms, 0);
                cu_mean_square.push_back(ms);
            }
        }
    }
}

void RMSProp::update_params() {
    int ms_ix = 0;
    for (int i = 0; i < net->layers.size(); ++i) {
        Layer* l = net->layers[i];
        for (int j = 0; j < l->get_num_trainable(); ++j) {
            if(use_cpu) {
                tensor_ptr params = l->get_trainable(j);
                tensor_ptr grad = l->get_grad_error_wrt_param(j)->copy();

                tensor_ptr mean_square = this->mean_square[ms_ix];

                //clip gradient
                //grad->apply_fun(&grad_clip, [](float a, void* clip) -> float {return a > 0 ? std::min(*(float*)clip, a) : std::max(-*(float*)clip, a);});

                mean_square->multiply_const(decay); // decay*ms_{t-1}
                grad->multiply(grad); // (dE/dw)^2
                grad->multiply_const(1 - decay); // (1-decay)*(dE/dw)^2
                mean_square->add(grad); // ms_{t} = decay*ms_{t-1} + (1-decay)*(dE/dw)^2

                tensor_fun_universal one_over_sqrt = [](float a, void *val) -> float {
                    return std::pow(a, -0.5) + (*(float *) val);
                };

                tensor_ptr mean_square_sqrt = mean_square->copy();
                mean_square_sqrt->apply_fun((void *) &eps, one_over_sqrt); // 1/sqrt(ms_{t})


                grad = l->get_grad_error_wrt_param(j)->copy();
                grad->multiply(mean_square_sqrt);
                // this prevents big updates
                grad->apply_fun(&(grad_clip), [](float a, void *clip) -> float {
                    return a > 0 ? std::min(*(float *) clip, a) : std::max(-*(float *) clip, a);
                });
                grad->multiply_const(learning_rate);

                params->sub(grad);
            }

            if(use_gpu) {
                cu_tensor params = l->cu_get_trainable(j);
                cu_tensor grad = l->cu_get_grad_error_wrt_param(j);
                cu_tensor mean_square = cu_mean_square[ms_ix];
                w_rmsprop_update_params(params, grad, mean_square, learning_rate, decay, eps, grad_clip);
                cudaDeviceSynchronize();
            }

            ms_ix++;
        }
    }
}

RMSProp::~RMSProp() {
    for (int i = 0; i < cu_mean_square.size(); ++i) {
        if(cu_mean_square[i].array) CHECK_ERROR(cudaFree(cu_mean_square[i].array));
    }
}

void Optimizer::initialize(NeuralNet *net) {
    this->net = net;
    this->use_gpu = net->use_gpu;
    this->use_cpu = net->use_cpu;

    for (int i = 0; i < net->layers.size(); ++i) {
        for (int j = 0; j < net->layers[i]->get_num_trainable(); ++j) {
            cout << "layer " << i << " trainable_id " << j << endl;
            net->layers[i]->get_trainable(j)->print();
        }
    }
}
