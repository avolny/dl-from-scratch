//
// Created by adam on 26.10.17.
//

#ifndef PROJECT_NEURALNET_H
#define PROJECT_NEURALNET_H

//#include "tensor/matrix.h"
#include "tensor/tensor.h"
#include "layers.h"
#include "neuralnet_gpu.h"
#include <vector>

using namespace std;

struct train_info {
    float loss;
    float accuracy;
};
//class Layer;
class Optimizer;

/**
 * A main class of this framework, this is the one which creates the neural network and provides an interface
 * to add layers, train it and use it for predictions
 */
class NeuralNet {
    friend class Optimizer;
    friend class SGD;
    friend class RMSProp;
public:
    /// Main constructor
    /// \param sequential set this to true if the model you're using is a linear graph (each layer has 1 predecessor and 1 succesor)
    NeuralNet(bool sequential);
    ~NeuralNet();
    /// Use this function to acquire an output for a single batch
    /// \param batch list of batch input components (1 for single input layer)
    /// \param n_batch_ins number of components
    /// \return list of output components (1 for single output layer)
    tensor_ptr* predict(tensor_ptr* batch, int n_batch_ins);
    /// Predict function for sequention models with single input layer and single output layer
    /// \param batch
    /// \return prediction
    tensor_ptr predict(tensor_ptr batch); //only for sequential models
    /// The function that's used to run the training of the model
    /// \param x a list of input batches, each batch is composed of components (1 for single input layer)
    /// \param y a list of desired outputs for each batch, also has components (1 for single output layer)
    /// \param n_batches a total number of batches
    /// \param n_batch_ins a number of components in input batch
    /// \param n_batch_outs a number of components in target batch
    /// \param n_epochs the number of epochs to train the model for (1 epoch = 1 iteration over all batches)
    void fit(tensor_ptr** x, tensor_ptr** y, int n_batches, int n_batch_ins, int n_batch_outs, int n_epochs);
    /// A function which can be used to numerically check correctness of gradients. Currently works only with cpu
    /// \param x a single input batch, containts input components (1 for single input layer)
    /// \param y a single desired output batch, contains desired output components (1 for single output layer)
    /// \param n_batch_ins number of components in input batch
    /// \param n_batch_outs number of components in target batch
    void validate_gradients(tensor_ptr* x, tensor_ptr* y, int n_batch_ins, int n_batch_outs);
    /// Adds a layer to the network, in case of sequential model, it is automatically appended after the last previous layer
    /// \param l
    void add_layer(Layer* l);
    /// Adds a layer with the option to mark the layer as an output one, in case of sequential model, it is automatically appended after the last previous layer
    /// \param l
    /// \param is_output true for output layer
    void add_layer(Layer* l, bool is_output);
    /// Adds layer as an output, in case of sequential model, it is automatically appended after the last previous layer
    /// \param l
    void add_output(Layer* l);
    /// Adds loss layer to the network, that is used for deciding the error function which is going to be minimized during training, in case of sequential model, it is automatically appended after the last previous layer
    /// \param l
    void add_loss_layer(LossLayer* l);
    /// Initialize the network for training, you can enable both gpu and cpu computation this can be used to compare the gradients of both and check their correctness, otherwise select only one
    /// \param batch_size number of samples in each batch
    /// \param optimizer an optimizer to be used for training
    /// \param use_gpu true if you want to compute on gpu
    /// \param use_cpu true if you want to compute on cpu
    void initialize(int batch_size, Optimizer* optimizer, bool use_gpu, bool use_cpu);
private:
    void train_batch(train_info &info, tensor_ptr *batch_x, tensor_ptr *batch_y, int n_batch_ins, int n_batch_outs);
    void reset_computation();
    void compute_output(bool use_loss);
    void compute_grad();
    void set_input(tensor_ptr *batch_x, int n_batch_ins);
    void set_target(tensor_ptr *batch_y, int n_batch_outs);
    float get_loss();
    bool initialized = false;
    int batch_size;
    bool sequential;
    bool use_gpu;
    bool use_cpu;
    Optimizer* optimizer;
    vector<Layer*> layers;
    vector<Layer*> output_layers;
    vector<InputLayer*> input_layers;
    vector<LossLayer*> loss_layers;
    vector<LossLayer*> loss_layers_supervised; // those require target values
};

/**
 * Optimizer is an object which takes precomputed gradients w.r.t. the parameters of the model, and updates those parameters according to those gradients
 */
class Optimizer {
public:
    /// This method contains the initialization of all variables the Optimizer is going to be using
    /// \param net the neural network which is using the optimizer, it's used to access the gradients and parameters later on
    virtual void initialize(NeuralNet* net);
    /// This method is called after each batch is presented and the gradients computed, it contains the update of the parameters
    virtual void update_params() = 0;
protected:
    NeuralNet* net;
    bool use_gpu;
    bool use_cpu;
};

/**
 * Stochastic Gradient Descent optimizer
 */
class SGD : public Optimizer {
public:
    /// Main constructor
    /// \param learning_rate decides magnitude of the parameter updates
    /// \param grad_clip decides maximum absolute value each gradient can reach, helps stability, recommended value around 1.0
    SGD(float learning_rate, float grad_clip);
    void initialize(NeuralNet* net);
    void update_params();
private:
    float learning_rate, grad_clip;
};

/**
 * RMSProp optimizer, see https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf for details
 * it is a first order method with momentum
 */
class RMSProp : public Optimizer {
public:
    /// Main constructor
    /// \param learning_rate decides magnitude of the parameter updates
    /// \param decay decides the ratio of how much the current gradient is used vs. how much the built up momentum is used, recommended value 0.9
    /// \param eps prevents zero division, recommended value of 1e-5
    /// \param grad_clip decides maximum absolute value each gradient can reach, helps stability, recommended value around 1.0
    RMSProp(float learning_rate, float decay, float eps, float grad_clip);
    ~RMSProp();
    void initialize(NeuralNet* net);
    void update_params();
private:
    float learning_rate, decay, eps, grad_clip;
    vector<tensor_ptr> mean_square;
    vector<cu_tensor> cu_mean_square;
};


#endif //PROJECT_NEURALNET_H
