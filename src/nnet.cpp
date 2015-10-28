#include <cmath> 
#include <iostream>
#include <stdexcept>
#include <ctime>
#include <cerrno>
#include <cfenv>

#include "nnet.h"
#include "../utils/auc.h"
#include "../utils/Eigen_plus.h"

#define TEST 0

using namespace neural_network;

nnet::nnet(VectorXi size_layers) :  eval("accuracy"),
                                    mode_test(0),
                                    random_state(-1), 
                                    verbose(1), 
                                    loss("logloss"), 
                                    batchsize(1), 
                                    epochs(1), 
                                    learning_rate(0.1), 
                                    Lambda(0)
                                    {
    this->size_layers = size_layers;        
    this->nn_size = size_layers.size();   
    init_internal_matrices();
}

nnet::~nnet(){
}

void nnet::init_internal_matrices(){

    if(random_state != -1)
        srand(random_state); //seed random number function
    else
        srand(time(0));        

    // initialize vector of matrices W using a random initialization
    W.resize(nn_size-1);
    for(unsigned int i=0; i<W.size(); i++){
        MatrixXd m(size_layers[i]+1, size_layers[i+1]);
        for(unsigned int j=0; j<m.rows(); j++){
            for(unsigned int k=0; k<m.cols(); k++){
                m(j,k) = ((double)rand() / (RAND_MAX + 1) - 0.5)*2.0*4.0*std::sqrt(6.0/(m.rows()+m.cols()));
            }
        }    
        W[i] = m;
    }
    
    // initialize vector of matrices A
    A.resize(nn_size);
    for(unsigned int i=0; i<A.size(); i++){
        if(i == nn_size-1){
            MatrixXd m(batchsize, size_layers[i]);
            A[i] = m;
        }else{
            MatrixXd m(batchsize, size_layers[i]+1);
            A[i] = m;
        }
    }
    
    // initialize vector of matrices Z
    Z.resize(nn_size);
    for(unsigned int i=0; i<nn_size; i++){
        if(i == nn_size-1){
            MatrixXd m(batchsize, size_layers[i]);
            Z[i] = m;
        }else{
            MatrixXd m(batchsize, size_layers[i]+1);
            Z[i] = m; 
        }
    }
    
    // initialize vector of matrices delta
    delta.resize(nn_size-1);
    for(unsigned int i=0; i<nn_size-1; i++){
        MatrixXd m(batchsize, size_layers[i+1]);          
        delta[i] = m;
    }
    
    // initialize vector of matrices dEdW
    dEdW.resize(nn_size-1);
    for(unsigned int i=0; i<nn_size-1; i++){
        MatrixXd m(size_layers[i]+1, size_layers[i+1]);             
        dEdW[i] = m;
    }
}

MatrixXd nnet::forward(MatrixXd &X, MatrixXd &y){

    // add bias term to the input matrix X
    MatrixXd bias = MatrixXd::Ones(X.rows(),1);    
    Z[0] = concatenate(bias, X, 1); 
    
    for(unsigned int i=0; i<nn_size-2; i++){
        // Compute activations for hidden layers
        A[i] =  Z[i]*W[i];   
        bias = MatrixXd::Ones(A[i].rows(), 1);               
        Z[i+1] = concatenate(bias, sigmoid(A[i]), 1);  
    }
    
    // compute activation of the last layer
    A[A.size()-1] =  Z[Z.size()-2]*W[W.size()-1];
    Z[Z.size()-1] = sigmoid( A[A.size()-1] );

    return Z[Z.size()-1];
}

void nnet::backpropagation(MatrixXd &X, MatrixXd &y){
    
    // get prediction Z[-1]
    forward(X, y);  
    
    if(loss.compare("quadratic")==0){
        MatrixXd err = Z[Z.size()-1] - y;
        MatrixXd dZdA = sigmoid_prime(Z[Z.size()-1]);  
        delta[delta.size()-1] = err.cwiseProduct(dZdA); 
    }else if(loss.compare("logloss")==0){   
        delta[delta.size()-1] = Z[Z.size()-1] - y;
    }else{
        throw std::invalid_argument("Invalid loss name.");
    }
    
    // compute deltas
    for(unsigned int i=nn_size-2; i>0; i--){                         
        MatrixXd Wt = W[i].transpose();
        Wt = slice(Wt, 0, Wt.rows(), 1, Wt.cols());
        delta[i-1] = delta[i] * Wt;
        MatrixXd Zb = slice(Z[i], 0, Z[i].rows(), 1, Z[i].cols());
        delta[i-1] = delta[i-1].cwiseProduct( sigmoid_prime( Zb ) );       
    }

    // compute gradient
    for(unsigned int i=0; i<nn_size-1; i++){
        dEdW[i] = Z[i].transpose() * delta[i];
        dEdW[i] = dEdW[i] / delta[i].rows();
        dEdW[i] = dEdW[i] + Lambda * W[i];
        // apply gradient
        W[i] = W[i] - learning_rate * dEdW[i]; 
    }

}    

double nnet::evaluation(MatrixXd &X, MatrixXd &y){  

    // predict labels
    MatrixXd preds = predict(X);
    
    // compute cost
    if(eval.compare("quadratic")==0){
        double cost = 0;
        MatrixXd m = preds - y;
        for(unsigned int i=0; i<m.rows(); i++){
            for(unsigned int j=0; j<m.cols(); j++){
                cost += 0.5*std::pow(m(i,j), 2);
            }
        }
        return cost/(y.rows()*y.cols());       
    }else if(eval.compare("logloss")==0){
        double cost = 0;
        double eps = 1e-15;
        for(unsigned int i=0; i<preds.rows(); i++){
            for(unsigned int j=0; j<preds.cols(); j++){
                if(preds(i,j) < eps){
                    if(y(i,j) == 0){
                        cost += - std::log(1.0-eps);
                    }else if(y(i,j)==1){
                        cost += - std::log(eps);
                    }else{
                        cost += - y(i,j)*std::log(eps) - (1.0-y(i,j))*std::log(1.0-eps);
                    }
                }else if(preds(i,j) > (1.0-eps) ){
                    if(y(i,j) == 0){
                        cost +=  - std::log(eps);
                    }else if (y(i,j)==1){
                        cost += - std::log(1.0-eps);
                    }else{
                        cost += - y(i,j)*std::log(1.0-eps) - (1.0-y(i,j))*std::log(eps);
                    }
                }else{
                    if(y(i,j) == 0){
                        cost += - std::log(1.0-preds(i,j));
                    }else if (y(i,j)==1){
                        cost += - std::log(preds(i,j));
                    }else{
                        cost += - y(i,j)*std::log(preds(i,j)) - (1.0-y(i,j))*std::log(1.0-preds(i,j));
                    }
                }
            }
        }
        return cost/(y.rows()*y.cols()); 
    }else if(eval.compare("accuracy")==0){
        double acc = 0;
        for(unsigned int i=0; i<preds.rows(); i++){
            if( (preds(i,0)>0.5) == y(i,0))
                acc += 1;
        }
        return acc/(y.rows());
    }else if(eval.compare("auc")==0){
        return calcAUC(y, preds, 1);
    }else{
        throw std::invalid_argument("Invalid eval name.");
    }
}

MatrixXd nnet::sigmoid(MatrixXd x) const {
    for(unsigned int i=0; i<x.rows(); i++){
        for(unsigned int j=0; j<x.cols(); j++){
            x(i,j) = 1.0/(1.0+std::exp(-x(i,j)));
        }
    }
    return x;
}

MatrixXd nnet::sigmoid_prime(MatrixXd x) const {
    for(unsigned int i=0; i<x.rows(); i++){
        for(unsigned int j=0; j<x.cols(); j++){
            x(i,j) = x(i,j)*(1.0-x(i,j));
        }
    }
    return x;
}

void nnet::fit(MatrixXd &X, MatrixXd &y){

    if(verbose) std::cout << "Size X(" << X.rows() << "," << X.cols() << ")" << std::endl;
    if(verbose) std::cout << "Size y(" << y.rows() << "," << y.cols() << ")" << std::endl;
    
    // initialize matrices
    init_internal_matrices();

    if(verbose) std::cout << "Batchsize: " << batchsize << std::endl;
    if(X.rows() % batchsize != 0)
        throw std::invalid_argument("Batchsize is not a multiple of the train size.");
    unsigned int n_batches = static_cast<int>(X.rows()/batchsize);
    if(verbose) std::cout << "n_batches: " << n_batches << std::endl;
    
    for(unsigned int e=0; e<epochs; e++){
    
        // create an array of indices of length N = X.rows()
        VectorXi indices = VectorXi::LinSpaced(X.rows(), 0, X.rows());
        if(mode_test == 0) // shuffle the indices
            std::random_shuffle(indices.data(), indices.data() + X.rows());
        
        // for all the batches
        for(unsigned int b=0; b<n_batches; b++){
            // slice the dataset using the shuffled indices
            MatrixXd batch_X = slice(X, indices, b*batchsize, (b+1)*batchsize);
            MatrixXd batch_y = slice(y, indices, b*batchsize, (b+1)*batchsize);
            
            // predict->backprop->apply gradient
            backpropagation(batch_X, batch_y);
        }

        // print roc auc
        if(verbose){
            std::cout << "Epoch:" << e << ", " << eval << " train: " << evaluation(X, y) << std::endl;    
        }
    
    }   
}

void nnet::fit(MatrixXd &X, MatrixXd &y, MatrixXd &X_val, MatrixXd &y_val){
        
    if(verbose) std::cout << "Size X(" << X.rows() << "," << X.cols() << ")" << std::endl;
    if(verbose) std::cout << "Size y(" << y.rows() << "," << y.cols() << ")" << std::endl;
    
    // initialize matrices
    init_internal_matrices();

    if(verbose) std::cout << "batchsize: " << batchsize << std::endl;
    if(X.rows() % batchsize != 0)
        throw std::invalid_argument("Batchsize is not a multiple of the train size.");
    unsigned int n_batches = static_cast<int>(X.rows()/batchsize);
    if(verbose) std::cout << "n_batches: " << n_batches << std::endl;
    
    for(unsigned int e=0; e<epochs; e++){
    
        // create an array of indices of length N = X.rows()
        VectorXi indices = VectorXi::LinSpaced(X.rows(), 0, X.rows());
        if(mode_test == 0) // shuffle the indices
            std::random_shuffle(indices.data(), indices.data() + X.rows());
        
        for(unsigned int b=0; b<n_batches; b++){
            // slice the dataset using the shuffled indices
            MatrixXd batch_X = slice(X, indices, b*batchsize, (b+1)*batchsize);
            MatrixXd batch_y = slice(y, indices, b*batchsize, (b+1)*batchsize);

            // predict->backprop->apply gradient
            backpropagation(batch_X, batch_y);
        }

        // print roc auc
        if(verbose){
            std::cout << "Epoch:" << e << ", " << eval << " train: " << evaluation(X, y);
            std::cout << ", " << eval <<  " val: " << evaluation(X_val, y_val) << std::endl;     
        }
    
    }   
}

MatrixXd nnet::predict(MatrixXd &X){

    // add bias term to the input matrix X
    MatrixXd bias = MatrixXd::Ones(X.rows(), 1);
    MatrixXd z = concatenate(bias, X, 1); 
    
    for(unsigned int i=0; i<nn_size-2; i++){
        z = sigmoid(z * W[i]);        
        bias = MatrixXd::Ones(z.rows(), 1); 
        z = concatenate(bias, z, 1);   
    }
    
    // compute activation of the last layer
    z = sigmoid( z * W[W.size()-1] );

    return z;
}


// Setters
void nnet::set_lambda(double Lambda){
    this->Lambda = Lambda;
}
void nnet::set_learning_rate(double learning_rate){
    this->learning_rate = learning_rate;
}
void nnet::set_epochs(unsigned int epochs){
    this->epochs = epochs;
}
void nnet::set_batchsize(unsigned int batchsize){
    this->batchsize = batchsize;
}
void nnet::set_loss(string loss){
    this->loss = loss;
}
void nnet::set_verbose(bool verbose){
    this->verbose = verbose;
}
void nnet::set_random_state(int random_state){
    this->random_state = random_state;
}
void nnet::set_W(VectorMatrixXd W){
    this->W = W;
}
void nnet::set_mode_test(bool test){
    this->mode_test = test;
}
void nnet::set_eval(string eval){
    this->eval = eval;
}

// Getters
VectorMatrixXd nnet::get_W(){
    return this->W;
}
VectorMatrixXd nnet::get_Z(){
    return this->Z;
}
VectorMatrixXd nnet::get_A(){
    return this->A;
}
VectorMatrixXd nnet::get_delta(){
    return this->delta;
}
VectorMatrixXd nnet::get_dEdW(){
    return this->dEdW;
}





