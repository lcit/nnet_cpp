#include <iostream>
#include "nnet.h"

using namespace neural_network;

int main(int argc, char** argv){



    MatrixXd X(4,2);
    X << 0,  0,
         0,  1, 
         1,  0,  
         1,  1;
    std::cout << "X------------" << std::endl;
    std::cout << X << std::endl;

    MatrixXd X_val(4,2);
    X_val << 0.2,  0,
         0,  1.2, 
         1,  0.1,  
         1.1,  1;
    std::cout << "X_val------------" << std::endl;
    std::cout << X_val << std::endl;    
     
    MatrixXd y(4,1);
    y << 1.0,
         0.0,
         0.0,
         1.0;    
    std::cout << "y------------" << std::endl;
    std::cout << y << std::endl;  

    MatrixXd y_val(4,1);
    y_val << 1.0,
         0.0,
         0.0,
         1.0;    
    std::cout << "y_val------------" << std::endl;
    std::cout << y_val << std::endl; 
    
    VectorXi layers(3);
    layers << 2,3,1;
    std::cout << "size_layers------------" << std::endl;
    std::cout << layers << std::endl; 
    
    nnet NN(layers); 
    NN.set_verbose(true);
    NN.set_epochs(100);
    NN.set_batchsize(1);
    NN.set_lambda(0.000001);
    NN.set_random_state(99);
    NN.set_learning_rate(0.5);
    NN.set_loss("logloss");   
    NN.set_eval("auc");
    NN.fit(X, y, X_val, y_val);
    
    std::cout << "X------------" << std::endl;
    std::cout << X << std::endl;
    std::cout << "y------------" << std::endl;
    std::cout << y << std::endl; 
    std::cout << "prediction------------" << std::endl;
    std::cout << NN.predict(X) << std::endl;
    
    std::cout << "Weights[0]------------" << std::endl;
    std::cout << NN.get_W()[0] << std::endl; 
    std::cout << "Weights[1]------------" << std::endl;
    std::cout << NN.get_W()[1] << std::endl;  
   

    return 0;
}