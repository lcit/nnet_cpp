# c++ class interface

from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool

cdef extern from "../wrapper/nnet_interface.h" namespace "neural_network":
    cdef cppclass nnet_int:
        # public
        nnet_int(vector[int]) except+
        void fit_int(vector[vector[double]], vector[vector[double]]) except+
        void fit_int(vector[vector[double]], vector[vector[double]], vector[vector[double]], vector[vector[double]]) except+
        vector[vector[double]] predict_int(vector[vector[double]])  except+
        void set_lambda_int(double)
        void set_learning_rate_int(double)
        void set_epochs_int(unsigned int)
        void set_batchsize_int(unsigned int)
        void set_loss_int(string)
        void set_verbose_int(bool)
        void set_random_state_int(int)
        void set_W_int(vector[vector[vector[double]]] W_int)
        void set_mode_test_int(bool)
        void set_eval_int(string)
        vector[vector[vector[double]]] get_W_int()
        vector[vector[vector[double]]] get_A_int()
        vector[vector[vector[double]]] get_Z_int()
        vector[vector[vector[double]]] get_delta_int()
        vector[vector[vector[double]]] get_dEdW_int()
        
  
# link the c function to python functions 
 
cdef class nnet:
    cdef nnet_int *thisptr    
    def __cinit__(self, size_layers):
        self.thisptr = new nnet_int(size_layers)
    def __dealloc__(self):
        del self.thisptr
    def fit(self, *args):
        if len(args) == 2:
            return self.thisptr.fit_int(args[0], args[1])
        elif len(args) == 4:
            return self.thisptr.fit_int(args[0], args[1], args[2], args[3])
    def predict(self, X):
        return self.thisptr.predict_int(X)
    def set_lambda(self, Lambda):
        self.thisptr.set_lambda_int(Lambda)
    def set_learning_rate(self, learning_rate):
        self.thisptr.set_learning_rate_int(learning_rate)
    def set_epochs(self, epochs):
        self.thisptr.set_epochs_int(epochs)
    def set_batchsize(self, batchsize):
        self.thisptr.set_batchsize_int(batchsize)
    def set_loss(self, loss):
        self.thisptr.set_loss_int(loss)
    def set_verbose(self, verbose):
        self.thisptr.set_verbose_int(verbose)
    def set_random_state(self, random_state):
        self.thisptr.set_random_state_int(random_state)
    def set_W(self, W):
        self.thisptr.set_W_int(W)
    def set_mode_test(self, test):
        return self.thisptr.set_mode_test_int(test)
    def set_eval(self, eval):
        self.thisptr.set_eval_int(eval)
    def get_W(self):
        return self.thisptr.get_W_int()
    def get_A(self):
        return self.thisptr.get_A_int()
    def get_Z(self):
        return self.thisptr.get_Z_int()
    def get_delta(self):
        return self.thisptr.get_delta_int()
    def get_dEdW(self):
        return self.thisptr.get_dEdW_int()
    