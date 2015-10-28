#include <cmath> 
#include <iostream>
#include <stdexcept>
#include <ctime>
#include <cerrno>
#include <cfenv>
#include <Eigen/Dense>

#include "nnet_interface.h"

using namespace neural_network;

// support function for conversion std MatrixD to Eigen MatrixXd
MatrixXd MatrixD_to_MatrixXd(MatrixD m){
    MatrixXd res(m.size(), m[0].size());   
    for (int i = 0; i < m.size(); i++)
        res.row(i) = VectorXd::Map(&m[i][0],m[i].size());
    return res;
}

// support function for conversion Eigen MatrixXd to std MatrixD
MatrixD MatrixXd_to_MatrixD(MatrixXd m){
    MatrixD res;   
    for (int i = 0; i < m.rows(); i++){
        VectorD v(m.row(i).data(), m.row(i).data() + m.row(i).rows() * m.row(i).cols());
        res.push_back(v);
    }
    return res;
}

nnet_int::nnet_int(VectorI size_layers_int){
    VectorXi size_layers = VectorXi::Map(&size_layers_int[0],size_layers_int.size());
    ptr_nnet = std::make_unique<nnet>(size_layers);
}

nnet_int::~nnet_int(){
}

void nnet_int::fit_int(MatrixD &X_int, MatrixD &y_int){

    MatrixXd X(X_int.size(), X_int[0].size());
    MatrixXd y(y_int.size(), y_int[0].size());
    for (int i = 0; i < X_int.size(); i++){
        X.row(i) = VectorXd::Map(&X_int[i][0],X_int[i].size());
        y.row(i) = VectorXd::Map(&y_int[i][0],y_int[i].size());
    }
    ptr_nnet->fit(X,y);
}

void nnet_int::fit_int(MatrixD &X_int, MatrixD &y_int, MatrixD &X_val_int, MatrixD &y_val_int){

    MatrixXd X(X_int.size(), X_int[0].size());
    MatrixXd y(y_int.size(), y_int[0].size());
    for (int i = 0; i < X_int.size(); i++){
        X.row(i) = VectorXd::Map(&X_int[i][0],X_int[i].size());
        y.row(i) = VectorXd::Map(&y_int[i][0],y_int[i].size());
    }
    MatrixXd X_val(X_val_int.size(), X_val_int[0].size());
    MatrixXd y_val(y_val_int.size(), y_val_int[0].size());
    for (int i = 0; i < X_val_int.size(); i++){
        X_val.row(i) = VectorXd::Map(&X_val_int[i][0],X_val_int[i].size());
        y_val.row(i) = VectorXd::Map(&y_val_int[i][0],y_val_int[i].size());
    }
        
    ptr_nnet->fit(X,y, X_val, y_val);  
}


MatrixD nnet_int::predict_int(MatrixD &X_int){

    MatrixXd X = MatrixD_to_MatrixXd(X_int);

    MatrixXd res = ptr_nnet->predict(X);
    
    MatrixD res_int = MatrixXd_to_MatrixD(res);
    
    return res_int;
}


// Setters
void nnet_int::set_lambda_int(double Lambda_int){
    ptr_nnet->set_lambda(Lambda_int);
}
void nnet_int::set_learning_rate_int(double learning_rate_int){
    ptr_nnet->set_learning_rate(learning_rate_int);
}
void nnet_int::set_epochs_int(unsigned int epochs_int){
    ptr_nnet->set_epochs(epochs_int);
}
void nnet_int::set_batchsize_int(unsigned int batchsize_int){
    ptr_nnet->set_batchsize(batchsize_int);
}
void nnet_int::set_loss_int(string loss_int){
    ptr_nnet->set_loss(loss_int);
}
void nnet_int::set_verbose_int(bool verbose_int){
    ptr_nnet->set_verbose(verbose_int);
}
void nnet_int::set_random_state_int(int random_state_int){
    ptr_nnet->set_random_state(random_state_int);
}
void nnet_int::set_mode_test_int(bool test_int){
    ptr_nnet->set_mode_test(test_int);
}
void nnet_int::set_eval_int(string eval_int){
    ptr_nnet->set_eval(eval_int);
}

// Getters
void nnet_int::set_W_int(VectorMatrixD W_int){
    VectorMatrixXd W;
    for (int i = 0; i < W_int.size(); i++){
        MatrixXd m = MatrixD_to_MatrixXd(W_int[i]);
        W.push_back(m);
    }
    ptr_nnet->set_W(W);
}

VectorMatrixD nnet_int::get_W_int(){
    VectorMatrixXd W = ptr_nnet->get_W();
    VectorMatrixD W_int;
    for (int i = 0; i < W.size(); i++){       
        MatrixD m = MatrixXd_to_MatrixD(W[i]);    
        W_int.push_back(m);
    }
 
    return W_int;
}

VectorMatrixD nnet_int::get_Z_int(){
    VectorMatrixXd Z = ptr_nnet->get_Z();
    VectorMatrixD Z_int;
    for (int i = 0; i < Z.size(); i++){       
        MatrixD m = MatrixXd_to_MatrixD(Z[i]);    
        Z_int.push_back(m);
    } 
    return Z_int;
}

VectorMatrixD nnet_int::get_A_int(){
    VectorMatrixXd A = ptr_nnet->get_A();
    VectorMatrixD A_int;
    for (int i = 0; i < A.size(); i++){       
        MatrixD m = MatrixXd_to_MatrixD(A[i]);    
        A_int.push_back(m);
    } 
    return A_int;
}

VectorMatrixD nnet_int::get_delta_int(){
    VectorMatrixXd delta = ptr_nnet->get_delta();
    VectorMatrixD delta_int;
    for (int i = 0; i < delta.size(); i++){       
        MatrixD m = MatrixXd_to_MatrixD(delta[i]);    
        delta_int.push_back(m);
    } 
    return delta_int;
}

VectorMatrixD nnet_int::get_dEdW_int(){
    VectorMatrixXd dEdW = ptr_nnet->get_dEdW();
    VectorMatrixD dEdW_int;
    for (int i = 0; i < dEdW.size(); i++){       
        MatrixD m = MatrixXd_to_MatrixD(dEdW[i]);    
        dEdW_int.push_back(m);
    } 
    return dEdW_int;
}
