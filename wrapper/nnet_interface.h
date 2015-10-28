// this class is just an interface to the real nnet class.
// Cython do not handle yet Eigen matrices, therefore, we
// convert the Eigen matrices here to std vector of vectors
// in order to properly create the python wrapper.

#ifndef NNET_INTERFACE_H
#define NNET_INTERFACE_H

#include "../src/nnet.h"
#include <vector>
#include <memory>
#include <Eigen/Dense>

using namespace Eigen;

typedef std::vector<int> VectorI;
typedef std::vector<VectorI> MatrixI;
typedef std::vector<double> VectorD;
typedef std::vector<VectorD> MatrixD;
typedef std::vector<MatrixXd> VectorMatrixXd;
typedef std::vector<MatrixD> VectorMatrixD;

namespace neural_network{

    class nnet_int{

        private:
                std::unique_ptr<nnet> ptr_nnet;      
        public:
            nnet_int(VectorI size_layers_int);
            ~nnet_int();
            void fit_int(MatrixD &X_int, MatrixD &y_int);
            void fit_int(MatrixD &X_int, MatrixD &y_int, MatrixD &X_val_int, MatrixD &y_val_int);
            MatrixD predict_int(MatrixD &X_int);
            void set_lambda_int(double Lambda_int);
            void set_learning_rate_int(double learning_rate_int);
            void set_epochs_int(unsigned int epochs_int);
            void set_batchsize_int(unsigned int batchsize_int);
            void set_loss_int(string loss_int);
            void set_verbose_int(bool verbose_int);
            void set_random_state_int(int random_state_int);
            void set_W_int(VectorMatrixD W_int);
            void set_mode_test_int(bool test_int);
            void set_eval_int(string eval_int);
            VectorMatrixD get_W_int();
            VectorMatrixD get_A_int();
            VectorMatrixD get_Z_int();
            VectorMatrixD get_delta_int();
            VectorMatrixD get_dEdW_int();            
    };
};
        
#endif