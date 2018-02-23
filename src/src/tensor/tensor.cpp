//
// Created by adam on 8.11.17.
//

#include "tensor.h"

#define MAX_PRINT_LEN 8
#define MAX_PRINT_ROWS 10

t_shape::t_shape(int *dims, int n_dims) {
    this->dims = dims;
    this->n_dims = n_dims;
}

t_shape::t_shape(const initializer_list<int> &il) {
    this->dims = new int[il.size()];
    this->n_dims = il.size();
    for (int i = 0; i < il.size(); ++i) {
        dims[i] = *(il.begin()+i);
    }
}

bool t_shape::eq(t_shape s) {
    if(this->n_dims != s.n_dims)
        return false;

    for (int i = 0; i < this->n_dims; ++i) {
        if(this->dims[i] != s.dims[i])
            return false;
    }
    return true;
}

bool t_shape::eq(t_shape *s) {
    if(this->n_dims != s->n_dims)
        return false;

    for (int i = 0; i < this->n_dims; ++i) {
        if(this->dims[i] != s->dims[i])
            return false;
    }
    return true;
}

bool t_shape::eq(cu_shape s) {
    t_shape* s2 = from_cu_shape(s);
    return eq(s2);
}

void t_shape::print() {
    cout << "shape: (";
    for (int i = 0; i < this->n_dims; ++i) {
        cout << this->dims[i];
        if(i < this->n_dims-1)
            cout << ", ";
    }
    cout << ")" << endl;
}

t_shape::~t_shape() {
    delete this->dims;
}

t_shape::t_shape(const t_shape &t) {
    this->n_dims = t.n_dims;
    this->dims = new int[t.n_dims];
    for (int i = 0; i < t.n_dims; ++i) {
        this->dims[i] = t.dims[i];
    }
}

t_shape::t_shape() {
    this->n_dims = 0;
    this->dims = nullptr;
}

int &t_shape::operator[](int index) {
    if(index < 0 || index >= this->n_dims) {
        cout << index;
        print();
        throw std::invalid_argument("the index of the shape dimension is out of bounds.");
    }
    return this->dims[index];
}

int t_shape::flat_size() {
    int size = 1;
    for (int i = 0; i < n_dims; ++i) {
        size *= dims[i];
    }
    return size;
}


bool tensor::dim_eq(tensor_ptr t) {
    return this->shape.eq(t->shape);
}

void tensor::force_dim_eq(tensor_ptr t) {
    if(!dim_eq(t)) {
        t->shape.print();
        shape.print();
        throw std::invalid_argument("tensors must have the same shape");
    }
}

void tensor::fill(float value) {
    tensor_fun_universal f = [](float a, void* val) -> float {
        return (*(float*)val);
    };

    float* v = new float;
    *v = value;
//    tensor_fun_unary f = [](float a) -> float {return a*1;};
    apply_fun((void*) v, f);
    delete v;
}

void tensor::add(tensor_ptr t) {
    force_dim_eq(t);
    this->apply_fun(t, [](float a, float b) -> float {return a+b;});
}

void tensor::sub(tensor_ptr t) {
    force_dim_eq(t);
    this->apply_fun(t, [](float a, float b) -> float {return a-b;});
}

void tensor::multiply(tensor_ptr t) {
    force_dim_eq(t);
    this->apply_fun(t, [](float a, float b) -> float {return a*b;});
}

void tensor::multiply_const(float val) {
    tensor_fun_universal f = [](float a, void* val) -> float {
        return a*(*(float*)val);
    };

    this->apply_fun((void*) &val, f);
}

void tensor::print(string caption) {
    cout << caption << endl;
    this->print();
}


tensor1d::tensor1d(int rows) : tensor1d(t_shape({rows})){

}

tensor_ptr tensor1d::dot(tensor_ptr t) {
    force_dim_eq(t);

    tensor1d_ptr t1 = std::dynamic_pointer_cast<tensor1d>(t);

    float sum = 0;
    for (int i = 0; i < t->shape[0]; ++i) {
        sum += t1->array[i]*this->array[i];
    }

    tensor1d_ptr t_sum = tensor1d_ptr(new tensor1d(1));
    t_sum->array[0] = sum;

    return std::static_pointer_cast<tensor>(t_sum);
}

tensor_ptr tensor1d::copy() {
    tensor1d_ptr t = tensor1d_ptr(new tensor1d(this->rows));
    for (int i = 0; i < t->rows; ++i) {
        t->array[i] = this->array[i];
    }
    return std::static_pointer_cast<tensor>(t);
}



void tensor1d::print() {
    cout << "tensor1d: {";
    for (int i = 0; i < std::min(MAX_PRINT_LEN, this->rows); ++i) {
        cout << this->array[i];
        if( i < std::min(MAX_PRINT_LEN, this->rows)-1)
            cout << ", ";
    }

    if (this->rows > MAX_PRINT_LEN) {
        cout << ", ... , ";
        for (int i = std::max(this->rows-3, MAX_PRINT_LEN); i < this->rows; ++i) {
            cout << this->array[i];
            if( i < this->rows-1)
                cout << ", ";
        }
    }

    cout << "}" << endl;
}

tensor_ptr tensor1d::reshape(t_shape s) {
    return nullptr;
}


tensor1d::~tensor1d() {
    delete this->array;
}

void tensor1d::apply_fun(tensor_fun_unary f) {
    for (int i = 0; i < this->rows; ++i) {
        this->array[i] = f(this->array[i]);
    }
}

void tensor1d::apply_fun(tensor_ptr t, tensor_fun_binary f) {
    force_dim_eq(t);

    tensor1d_ptr t1 = std::dynamic_pointer_cast<tensor1d>(t);

    for (int i = 0; i < this->rows; ++i) {
        this->array[i] = f(this->array[i], t1->array[i]);
    }
}

void tensor1d::apply_fun(void *p, tensor_fun_universal f) {
    for (int i = 0; i < this->rows; ++i) {
        this->array[i] = f(this->array[i], p);
    }
}

tensor1d::tensor1d(t_shape shape) : tensor(shape) {
    if(shape.n_dims != 1) {
        shape.print();
        throw std::invalid_argument( "trying to initialize 1d tensor with non-1d shape" );
    }

    this->rows = shape.dims[0];
    this->array = new float[rows];
    for (int i = 0; i < rows; ++i) {
        this->array[i] = 0;
    }
}

float tensor1d::sum() {
    float sum = 0;
    for (int i = 0; i < this->rows; ++i) {
        sum += this->array[i];
    }
    return sum;
}

tensor_ptr tensor1d::flatten() {
    return copy();
}

tensor_ptr tensor1d::transpose() {
    return copy();
}

tensor_ptr tensor1d::sum(int axis) {
    if(axis != 0) {
        cout << axis << endl;
        throw std::invalid_argument("you can sum only over axis 0 in tensor1d.");
    }
    tensor1d_ptr sum = tensor1d_ptr(new tensor1d(1));
    sum->array[0] = this->sum();
    return sum;
}

tensor_ptr tensor1d::repeat(int axis, int n_times) {
    if(axis != 0) {
        cout << axis << endl;
        throw std::invalid_argument("you can repeat only over axis 0 in tensor1d.");
    }

    tensor1d_ptr t = tensor1d_ptr(new tensor1d(n_times*rows));
    for (int i = 0; i < n_times; ++i) {
        for (int j = 0; j < rows; ++j) {
            t->array[i*rows + j] = array[j];
        }
    }

    return t;
}

float tensor1d::mean() {
    return sum()/rows;
}

float tensor1d::max() {
    float max = array[0];
    for (int i = 0; i < rows; ++i) {
        if(array[i] > max)
            max = array[i];
    }
    return max;
}

float tensor1d::min() {
    float min = array[0];
    for (int i = 0; i < rows; ++i) {
        if(array[i] < min)
            min = array[i];
    }
    return min;
}

tensor2d::tensor2d(int rows, int cols) : tensor2d(t_shape({rows, cols})) {

}


tensor_ptr tensor2d::dot(tensor_ptr t) {
    if(t->shape.n_dims != 2) {
        t->shape.print();
        throw std::invalid_argument( "you can only do a dot product with 2d tensor" );
    }

    if(this->shape.dims[1] != t->shape.dims[0]) {
        this->shape.print();
        t->shape.print();
        throw std::invalid_argument( "tensors have mismatching dimensions (the middle must be equal)" );
    }

    tensor2d_ptr t1 = std::dynamic_pointer_cast<tensor2d>(t);
    tensor2d_ptr result = tensor2d_ptr(new tensor2d(this->shape[0], t->shape[1]));

    for (int i = 0; i < result->rows; ++i) {
        for (int j = 0; j < result->cols; ++j) {
            result->array[i][j] = 0;
            for (int k = 0; k < this->cols; ++k) {
                result->array[i][j] += this->array[i][k]*t1->array[k][j];
            }
        }
    }

    return std::static_pointer_cast<tensor>(result);
}

tensor_ptr tensor2d::copy() {
    tensor2d_ptr t = tensor2d_ptr(new tensor2d(this->rows, this->cols));
    for (int i = 0; i < this->rows; ++i) {
        for (int j = 0; j < this->cols; ++j) {
            t->array[i][j] = this->array[i][j];
        }
    }

    return t;
}

tensor_ptr tensor2d::flatten() {
    tensor1d_ptr result = tensor1d_ptr(new tensor1d(this->rows*this->cols));
    for (int i = 0; i < this->rows; ++i) {
        for (int j = 0; j < this->cols; ++j) {
            result->array[i*this->cols + j] = this->array[i][j];
        }
    }

    return std::static_pointer_cast<tensor>(result);
}

tensor_ptr tensor2d::reshape(t_shape s) {
    if(s.n_dims != 2) {
        s.print();
        throw std::invalid_argument( "trying to reshape into 2d tensor, while providing shape with different # of dims." );
    }

    if(s.dims[0]*s.dims[1] != this->shape.dims[0]*this->shape.dims[1]) {
        s.print();
        this->shape.print();
        throw std::invalid_argument( "the provided shape does not contain the same number of elements as the original tensor" );
    }

    tensor2d_ptr result = tensor2d_ptr(new tensor2d(s.dims[0], s.dims[1]));
    tensor1d_ptr inter = std::dynamic_pointer_cast<tensor1d>(this->flatten());

    for (int i = 0; i < result->shape.dims[0]; ++i) {
        for (int j = 0; j < result->shape.dims[1]; ++j) {
            result->array[i][j] = inter->array[i*result->cols + j];
        }
    }

    return std::static_pointer_cast<tensor>(result);
}

void tensor2d::print() {
    cout << "tensor2d: {" << endl;
    for (int i = 0; i < std::min(MAX_PRINT_ROWS, this->rows); ++i) {
        cout << " {";
        for (int j = 0; j < std::min(MAX_PRINT_LEN, this->cols); ++j) {
            cout << this->array[i][j];
            if( j < std::min(MAX_PRINT_LEN, this->cols)-1)
                cout << ", ";
        }

        if (this->cols > MAX_PRINT_LEN) {
            cout << ", ... , ";
            for (int j = std::max(this->cols-3, MAX_PRINT_LEN); j < this->cols; ++j) {
                cout << this->array[i][j];
                if( j < this->cols-1)
                    cout << ", ";
            }
        }
        cout << "}" << endl;
    }

    if (this->rows > MAX_PRINT_ROWS) {
        cout << " ... " << endl << " ... " << endl << " ... " << endl;

        for (int i = std::max(MAX_PRINT_ROWS, this->rows-3); i < this->rows; ++i) {
            cout << " {";
            for (int j = 0; j < std::min(MAX_PRINT_LEN, this->cols); ++j) {
                cout << this->array[i][j];
                if( j < std::min(MAX_PRINT_LEN, this->cols)-1)
                    cout << ", ";
            }

            if (this->cols > MAX_PRINT_LEN) {
                cout << ", ... , ";
                for (int j = std::max(this->cols-3, MAX_PRINT_LEN); j < this->cols; ++j) {
                    cout << this->array[i][j];
                    if( j < this->cols-1)
                        cout << ", ";
                }
            }
            cout << "}" << endl;
        }
    }

    cout << "}" << endl;

}

tensor2d::~tensor2d() {
    for (int i = 0; i < this->rows; ++i) {
        delete this->array[i];
    }
    delete this->array;
}

void tensor2d::apply_fun(tensor_fun_unary f) {
    for (int i = 0; i < this->rows; ++i) {
        for (int j = 0; j < this->cols; ++j) {
            this->array[i][j] = f(this->array[i][j]);
        }
    }
}

void tensor2d::apply_fun(tensor_ptr t, tensor_fun_binary f) {
    tensor2d_ptr t1 = std::dynamic_pointer_cast<tensor2d>(t);

    for (int i = 0; i < this->rows; ++i) {
        for (int j = 0; j < this->cols; ++j) {
            this->array[i][j] = f(this->array[i][j], t1->array[i][j]);
        }
    }
}

void tensor2d::apply_fun(void *p, tensor_fun_universal f) {
    for (int i = 0; i < this->rows; ++i) {
        for (int j = 0; j < this->cols; ++j) {
            this->array[i][j] = f(this->array[i][j], p);
        }
    }
}


tensor2d::tensor2d(t_shape shape) : tensor(shape) {
    if(shape.n_dims != 2) {
        shape.print();
        throw std::invalid_argument( "trying to initialize 2d tensor with non-2d shape" );
    }

    rows = shape.dims[0];
    cols = shape.dims[1];

    this->array = new float*[rows];
    for (int i = 0; i < rows; ++i) {
        this->array[i] = new float[cols];

        for (int j = 0; j < cols; ++j) {
            this->array[i][j] = 0;
        }
    }

}

float tensor2d::sum() {
    float sum = 0;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            sum += array[i][j];
        }
    }

    return sum;
}

tensor_ptr tensor2d::transpose() {
    tensor2d_ptr t2 = tensor2d_ptr(new tensor2d(shape[1], shape[0]));
    for (int i = 0; i < shape[0]; ++i) {
        for (int j = 0; j < shape[1]; ++j) {
            t2->array[j][i] = array[i][j];
        }
    }

    return std::static_pointer_cast<tensor>(t2);
}

tensor_ptr tensor2d::sum(int axis) {
    if(axis > 1) {
        cout << axis << endl;
        throw std::invalid_argument("you can sum only over axis 0 or 1 in tensor2d.");
    }

    tensor2d_ptr t;
    if(axis == 0) {
        t = tensor2d_ptr(new tensor2d(1, shape[1]));
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                t->array[0][j] += array[i][j];
            }
        }
    } else {
        t = tensor2d_ptr(new tensor2d(shape[0], 1));
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                t->array[i][0] += array[i][j];
            }
        }
    }

    return t;
}

tensor_ptr tensor2d::repeat(int axis, int n_times) {
    if(axis > 1) {
        cout << axis << endl;
        throw std::invalid_argument("you can repeat only over axis 0 or 1 in tensor2d.");
    }

    tensor2d_ptr t;
    if(axis == 0) {
        t = tensor2d_ptr(new tensor2d(shape[0]*n_times, shape[1]));

        for (int k = 0; k < n_times; ++k) {
            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < cols; ++j) {
                    t->array[k*rows+i][j] = array[i][j];
                }
            }
        }
    } else {
        t = tensor2d_ptr(new tensor2d(shape[0], shape[1]*n_times));

        for (int k = 0; k < n_times; ++k) {
            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < cols; ++j) {
                    t->array[i][k*cols+j] = array[i][j];
                }
            }
        }
    }

    return t;
}

float tensor2d::mean() {
    return sum()/(rows*cols);
}

float tensor2d::max() {
    float max = array[0][0];
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if(array[i][j] > max)
                max = array[i][j];
        }
    }
    return max;
}

float tensor2d::min() {
    float min = array[0][0];
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if(array[i][j] < min)
                min = array[i][j];
        }
    }
    return min;
}


tensor_ptr new_tensor(t_shape shape) {
    if(shape.n_dims == 1)
        return tensor_ptr(new tensor1d(shape));
    else if(shape.n_dims == 2)
        return tensor_ptr(new tensor2d(shape));
    else {
        shape.print();
        throw std::invalid_argument("the provided shape must be either 1d or 2d");
    }

    return tensor_ptr();
}

tensor_ptr new_tensor(const initializer_list<int> &il) {
    return new_tensor(t_shape(il));
}

cu_shape to_cu_shape(t_shape s)  {
//    s.print();
    cu_shape c;
    c.n_dims = s.n_dims;
    c.len = 1;
    if(c.n_dims >= 1) {
        c.d1 = s.dims[0];
        c.len *= c.d1;
    }
    if(c.n_dims >= 2) {
        c.d2 = s.dims[1];
        c.len *= c.d2;
    }
    if(c.n_dims >= 3) {
        c.d3 = s.dims[2];
        c.len *= c.d3;
    }
    if(c.n_dims >= 4)
        throw std::runtime_error("cannot handle more than 3D tensors");

    return c;
}

t_shape *from_cu_shape(cu_shape s) {
    if(s.n_dims == 1) {
        return new t_shape({s.d1});
    } else if(s.n_dims == 2) {
        return new t_shape({s.d1, s.d2});
    } else if(s.n_dims == 3) {
        return new t_shape({s.d1, s.d2, s.d3});
    }
    return nullptr;
}

cu_tensor cu_alloc_tensor(t_shape shape) {
    cu_tensor t;
    t.shape = to_cu_shape(shape);
    CHECK_ERROR( cudaMalloc(&t.array, sizeof(float)*t.shape.len) );
    return t;
}

tensor_ptr tensor_from_cu(cu_tensor tensor) {
    float *data = new float[tensor.shape.len];
    CHECK_ERROR( cudaMemcpy(data, tensor.array, sizeof(float)*tensor.shape.len, cudaMemcpyDeviceToHost) );

    if(tensor.shape.n_dims == 1) {
        tensor1d_ptr tt = tensor1d_ptr(new tensor1d(tensor.shape.d1));
        for (int i = 0; i < tt->rows; ++i) {
            tt->array[i] = data[i];
        }
        return tt;
    } else if(tensor.shape.n_dims == 2) {
        tensor2d_ptr tt = tensor2d_ptr(new tensor2d(tensor.shape.d1, tensor.shape.d2));
        for (int i = 0; i < tt->rows; ++i) {
            for (int j = 0; j < tt->cols; ++j) {
                tt->array[i][j] = data[i*tt->cols + j];
            }
        }
        return tt;
    } else {
        throw runtime_error("NOT IMPLEMENTED");
    }

    return nullptr;
}

cu_tensor tensor_to_cu(tensor_ptr tensor) {
    tensor1d_ptr t = dynamic_pointer_cast<tensor1d>(tensor->flatten());
    cu_tensor cu_t = cu_alloc_tensor(tensor->shape);
    CHECK_ERROR( cudaMemcpy(cu_t.array, t->array, sizeof(float)*cu_t.shape.len, cudaMemcpyHostToDevice) );
    return cu_t;
}

tensor_ptr tensor2d_range(int rows, int cols) {
    tensor2d_ptr t = tensor2d_ptr(new tensor2d(rows, cols));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            t->array[i][j] = i*cols + j;
        }
    }

    return t;
}

tensor_ptr tensor1d_range(int rows) {
    tensor1d_ptr t = tensor1d_ptr(new tensor1d(rows));
    for (int i = 0; i < rows; ++i) {
        t->array[i] = i;
    }

    return t;
}

void cu_print_tensor(cu_tensor tensor, string caption) {
    cout << caption << "tensor, shape: ";
    cu_shape_print(tensor.shape);
    tensor_from_cu(tensor)->print();
}

void cu_print_tensor(cu_tensor tensor) {
    cu_print_tensor(tensor, "");
}



