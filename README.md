# Deep Learning From Scratch

This project has been created as a part of subject A4M39GPU at Czech Technical University. As the title suggests, it is a deep learning framework written from scratch, using C/C++ and pure CUDA (no cuDDD). The reason why I started this project was pure curiosity as I've always found a reasonable satisfaction in <i>re-inventing the wheel</i>. Experiencing an engineering problem entirely for oneself leads to a deep and intuitive understanding of the existing solutions and also it's great fun!

The project can be built using CMake, and the basic usage can be seen in [/src/src/main.cpp](/src/src/main.cpp). The API functions are designed to be used similarly as in Python frameworks Keras and NumPy as those are great examples of highly functional yet intuitive interfaces.

More detailed description of the framework and the design can be seen in the [report.pdf](/report.pdf).

Here is a snippet from the [/src/src/main.cpp](/src/src/main.cpp) demonstrating the basic usage.

```
tensor_ptr **x, **y; // x - input data, y - correct labels
int n_batches;
mnist_create_data(x, y, n_batches, 256, dataset.training_images, dataset.training_labels, true);

WeightInitializer* winit = new WeightInitializer(WeightInitializer::Type::random_uni); // initializes weights

int units = 128; // number of neurons in each hidden layer
int epochs = 25; // total number of iterations over dataset

NeuralNet* net = new NeuralNet(true); // true - sequential model
net->add_layer((Layer*) new InputLayer(new t_shape({784}))); // input layer
net->add_layer((Layer*) new DenseLayer(units, Activation::relu, winit)); // hidden layer
net->add_layer((Layer*) new DenseLayer(units, Activation::relu, winit)); // hidden layer
net->add_layer((Layer*) new DenseLayer(10, Activation::softmax, winit), true); // output layer
net->add_loss_layer((LossLayer*) new CrossEntropyError()); // loss function

RMSProp* rmsprop = new RMSProp(0.001, 0.99, 0.0001, 0.5); // RMSProp optimizer
net->initialize(256, rmsprop, true, false); // (batch size, optimizer, use_gpu, use_cpu)

net->fit(x, y, n_batches, 1, 1, epochs); // train the network on x,y with n_batches 
```

Special thanks go to the user <i>wichtounete</i>, whose code is used for loading the MNIST dataset, see the project repository [https://github.com/wichtounet/mnist](https://github.com/wichtounet/mnist)
