#include <iostream>
#include <thread>
#include <ctime>

#include "network/neuralnet.h"
#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"
#include <experimental/filesystem>
#include <chrono>

using namespace std;

void mnist_create_data(tensor_ptr** &x, tensor_ptr** &y, int &n_batches, int batch_size,
                       std::vector<std::vector<float>> data, std::vector<uint8_t> labels, bool shuffle) {
    int size = data.size();
    int data_len = 784;

    int* ix = new int[size];
    for (int l = 0; l < size; ++l) {
        ix[l] = l;
    }

    cout << size << " images in dataset" << endl;

    if(shuffle)
        std::random_shuffle ( ix, ix+size );

    n_batches = (int) std::floor(size*1./batch_size);

    x = new tensor_ptr*[n_batches];
    y = new tensor_ptr*[n_batches];

    int data_id = 0;
    for (int i = 0; i < n_batches; ++i) {
        x[i] = new tensor_ptr[1];
        y[i] = new tensor_ptr[1];
        x[i][0] = new_tensor(t_shape({data_len, batch_size}));
        y[i][0] = new_tensor(t_shape({10, batch_size}));
        tensor2d_ptr x2d = std::dynamic_pointer_cast<tensor2d>(x[i][0]);
        tensor2d_ptr y2d = std::dynamic_pointer_cast<tensor2d>(y[i][0]);

        for (int j = 0; j < batch_size; ++j) {
            if(data_id < n_batches*batch_size) {

                for (int k = 0; k < data_len; ++k) { // fill x
                    x2d->array[k][j] = data[ix[data_id]][k];
                }

                y2d->array[labels[ix[data_id]]][j] = 1; // fill one-hot y

                data_id++;
            }
        }
    }
}

int main_mnist_gpu() {
//    string path = "/home/adam/school/gpu/project/mnist";
    string path = "mnist";
    // MNIST_DATA_LOCATION set by MNIST cmake config
    std::cout << "MNIST data directory: " << path << std::endl;

    // Load MNIST data
    mnist::MNIST_dataset<std::vector, std::vector<float>, uint8_t> dataset =
            mnist::read_dataset<std::vector, std::vector, float, uint8_t>(path);
    std::cout << "first image " << dataset.training_images[0].size() << endl;

    mnist::normalize_dataset(dataset);

    tensor_ptr **x, **y;
    int n_batches;
    mnist_create_data(x, y, n_batches, 256, dataset.training_images, dataset.training_labels, true);


    WeightInitializer* winit = new WeightInitializer(WeightInitializer::Type::random_uni);

    int units = 128;
    int epochs = 25;

    NeuralNet* net = new NeuralNet(true);
    net->add_layer((Layer*) new InputLayer(new t_shape({784})));
    net->add_layer((Layer*) new DenseLayer(units, Activation::relu, winit));
    net->add_layer((Layer*) new DenseLayer(units, Activation::relu, winit));
    net->add_layer((Layer*) new DenseLayer(10, Activation::softmax, winit), true);
    net->add_loss_layer((LossLayer*) new CrossEntropyError());

    //    SGD* sgd = new SGD(0.0001, 1);
    RMSProp* rmsprop = new RMSProp(0.001, 0.99, 0.0001, 0.5);
    net->initialize(256, rmsprop, true, false);

    chrono::high_resolution_clock::time_point t1 = chrono::high_resolution_clock::now();

    net->fit(x, y, n_batches, 1, 1, epochs);

    chrono::high_resolution_clock::time_point t2 = chrono::high_resolution_clock::now();
    double dif = chrono::duration_cast<chrono::nanoseconds>( t2 - t1 ).count();
    printf ("Elasped time for %d epochs is %lf seconds.\n", epochs, dif/1e9 );

    return 0;
}

int main_xor_cpu() {
    srand(0);

    tensor2d_ptr input1 = tensor2d_ptr(new tensor2d(2,4));
    input1->array[0][0] = 0;
    input1->array[1][0] = 0;
    input1->array[0][1] = 0;
    input1->array[1][1] = 1;
    input1->array[0][2] = 1;
    input1->array[1][2] = 0;
    input1->array[0][3] = 1;
    input1->array[1][3] = 1;


    tensor2d_ptr target1 = tensor2d_ptr(new tensor2d(2,4));
    target1->array[0][0] = 0;
    target1->array[1][0] = 1;
    target1->array[0][1] = 1;
    target1->array[1][1] = 0;
    target1->array[0][2] = 1;
    target1->array[1][2] = 0;
    target1->array[0][3] = 0;
    target1->array[1][3] = 1;


    tensor_ptr** x = new tensor_ptr*[1];
    x[0] = new tensor_ptr[1];
    x[0][0] = input1;


    tensor_ptr** y = new tensor_ptr*[1];
    y[0] = new tensor_ptr[1];
    y[0][0] = target1;

    WeightInitializer* winit = new WeightInitializer(WeightInitializer::Type::random_uni);

    NeuralNet* net = new NeuralNet(true);
    net->add_layer((Layer*) new InputLayer(new t_shape({2})));
    net->add_layer((Layer*) new DenseLayer(5, Activation::sigmoid, winit));
    net->add_layer((Layer*) new DenseLayer(2, Activation::softmax, winit), true);
    net->add_loss_layer((LossLayer*) new MSE());

//    SGD* sgd = new SGD(0.0001, 1);
    RMSProp* rmsprop = new RMSProp(0.001, 0.9, 0.0001, 1);

    net->initialize(4, rmsprop, false, true);

    tensor_ptr out = net->predict(input1);
    out->print();

    net->fit(x, y, 1, 1, 1, 5000);
//    net->validate_gradients(x[0], y[0], 1, 1);

    out = net->predict(input1);
    cout << "network output: " << endl;
    out->print();
    cout << "correct output: " << endl;
    y[0][0]->print();

    return 0;
}

int main_xor_validate() {
    srand(0);

    tensor2d_ptr input1 = tensor2d_ptr(new tensor2d(2,4));
    input1->array[0][0] = 0;
    input1->array[1][0] = 0;
    input1->array[0][1] = 0;
    input1->array[1][1] = 1;
    input1->array[0][2] = 1;
    input1->array[1][2] = 0;
    input1->array[0][3] = 1;
    input1->array[1][3] = 1;


    tensor2d_ptr target1 = tensor2d_ptr(new tensor2d(2,4));
    target1->array[0][0] = 0;
    target1->array[1][0] = 1;
    target1->array[0][1] = 1;
    target1->array[1][1] = 0;
    target1->array[0][2] = 1;
    target1->array[1][2] = 0;
    target1->array[0][3] = 0;
    target1->array[1][3] = 1;


    tensor_ptr** x = new tensor_ptr*[1];
    x[0] = new tensor_ptr[1];
    x[0][0] = input1;


    tensor_ptr** y = new tensor_ptr*[1];
    y[0] = new tensor_ptr[1];
    y[0][0] = target1;

    WeightInitializer* winit = new WeightInitializer(WeightInitializer::Type::random_uni);

    NeuralNet* net = new NeuralNet(true);
    net->add_layer((Layer*) new InputLayer(new t_shape({2})));
    net->add_layer((Layer*) new DenseLayer(5, Activation::sigmoid, winit));
    net->add_layer((Layer*) new DenseLayer(2, Activation::softmax, winit), true);
    net->add_loss_layer((LossLayer*) new MSE());

    RMSProp* rmsprop = new RMSProp(0.001, 0.9, 0.0001, 1);

    net->initialize(4, rmsprop, false, true);

    net->validate_gradients(x[0], y[0], 1, 1);

//    cout << "abcd";

    return 0;
}

int main() {
    main_mnist_gpu();
//    main_xor_cpu();
//    main_xor_validate();
}

