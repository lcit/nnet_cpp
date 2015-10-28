#ifndef NNET_H
#define NNET_H

#include <cmath>
#include <stdio.h>
#include <memory>
#include <cstdlib>
#include <vector>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

typedef std::vector<MatrixXd> VectorMatrixXd;

namespace neural_network{

    class nnet{

        private:
            unsigned int    nn_size;        // number of layers
            VectorXi        size_layers;    // number of neuron per layer
            double          Lambda;         // Regularization term
            double          learning_rate;  // Learning rapidity
            unsigned int    epochs;         // number of iterations
            unsigned int    batchsize;      // online=1, 1<minibatch<N, fullbatch=N (N:training samples)
            string          loss;           // cost function ('quadratic', 'logloss')
            bool            verbose;        // print extra infos
            int             random_state;   // random seed
            VectorMatrixXd  W;              // vector of matrices, it contains the NN weights
            VectorMatrixXd  A;              // vector of matrices, it contains the NN input activation
            VectorMatrixXd  Z;              // vector of matrices, it contains the NN activation
            VectorMatrixXd  delta;          // vector of matrices, it contains the NN deltas
            VectorMatrixXd  dEdW;           // vector of matrices, it contains the NN gradients
            bool            mode_test;      
            string          eval;           // Evaluate and display a specific score ('quadratic loss', 'logloss', 'accuracy', 'auc')
            
            // Forward process
            // Parameters:
            //     X: train dataset (one batch)
            //     y: train labels (one batch)
            // Return:
            //     prediction of the labels y (Z[-1])
            MatrixXd forward(MatrixXd &X, MatrixXd &y);
            
            // Backpropagation process
            // Parameters:
            //     X: train dataset (one batch)
            //     y: train labels (one batch)
            // Return:
            //     
            void backpropagation(MatrixXd &X, MatrixXd &y);
            
            // Calculate the resulting cost or score
            // Parameters:
            //     X: train dataset (full dataset)
            //     y: train labels (full dataset)
            // Return:
            //     cost value
            double evaluation(MatrixXd &X, MatrixXd &y);

            // Sigmoid function
            // Parameters:
            //     x: matrix
            // Return:
            //     matrix
            MatrixXd sigmoid(MatrixXd x) const;
            
            // Derivative of sigmoid function
            // Parameters:
            //     x: matrix
            // Return:
            //     matrix
            MatrixXd sigmoid_prime(MatrixXd x) const;
            
            // Support function for NN initialization
            // Parameters:
            //     
            // Return:
            //     
            void init_internal_matrices();
        
        public:
            // Constructor
            // Parameters:
            //     size_layers: number of neurons per layer (i.e.:[2,3,1])
            nnet(VectorXi size_layers);
            ~nnet();
            
            // Fitting the NN
            // Parameters:
            //     X: full training set
            //     y: full training labels
            //     X_val: full validation set
            //     y_val: full validation labels
            // Return:
            //     
            void fit(MatrixXd &X, MatrixXd &y);
            void fit(MatrixXd &X, MatrixXd &y, MatrixXd &X_val, MatrixXd &y_val);
            
            // Predict labels of X, similar with forward() 
            // but no class members are modified
            // Parameters:
            //     X: full training set
            // Return:
            //     matrix of predictions
            MatrixXd predict(MatrixXd &X);
            
            // Setters
            void set_lambda(double Lambda);
            void set_learning_rate(double learning_rate);
            void set_epochs(unsigned int epochs);
            void set_batchsize(unsigned int batchsize);
            void set_loss(string loss);
            void set_verbose(bool verbose);
            void set_random_state(int random_state);
            void set_W(VectorMatrixXd W);
            void set_mode_test(bool test);
            void set_eval(string eval);
            
            
            // Getters
            VectorMatrixXd get_W();
            VectorMatrixXd get_A();
            VectorMatrixXd get_Z();
            VectorMatrixXd get_delta();
            VectorMatrixXd get_dEdW();           
    };
};

#endif