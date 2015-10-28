#include <ctime>
#include "Eigen_plus.h"


MatrixXd slice(const MatrixXd matrix, const int s1, const int e1){

    if(s1<0 || e1<0)
        throw std::invalid_argument("slice: Start or end positions are negative.");
    if(s1>e1)
        throw std::invalid_argument("slice: Start cannot be greater than end position.");
        
    MatrixXd m(e1-s1, matrix.cols());
    for(int i=0; i<e1-s1; i++){
        for(int j=0; j<matrix.cols(); j++){
            m(i,j) = matrix(i+s1,j);
        }
    }
    return m;
}
MatrixXd slice(const MatrixXd &matrix, const VectorXi indices, const int s1, const int e1){

    if(s1<0 || e1<0)
        throw std::invalid_argument("slice: Start or end positions are negative.");
    if(s1>e1)
        throw std::invalid_argument("slice: Start cannot be greater than end position.");
        
    MatrixXd m(e1-s1, matrix.cols());
    for(int i=0; i<e1-s1; i++){
        for(int j=0; j<matrix.cols(); j++){
            m(i,j) = matrix(indices[i+s1],j);
        }
    }
    return m;
}

MatrixXd slice(const MatrixXd &matrix, const int s1, const int e1, const int s2, const int e2){

    if(s1<0 || s2<0 || e1<0 || e2<0)
        throw std::invalid_argument("slice: Start or end positions are negative.");
    if(s1>e1 || s2>e2)
        throw std::invalid_argument("slice: Start cannot be greater than end position.");
        
    MatrixXd m(e1-s1, e2-s2);
    for(int i=0; i<e1-s1; i++){
        for(int j=0; j<e2-s2; j++){
            m(i,j) = matrix(i+s1,j+s2);
        }
    }
    return m;
}
MatrixXd slice(const MatrixXd &matrix, const VectorXi indices, const int s1, const int e1, const int s2, const int e2){

    if(s1<0 || s2<0 || e1<0 || e2<0)
        throw std::invalid_argument("slice: Start or end positions are negative.");
    if(s1>e1 || s2>e2)
        throw std::invalid_argument("slice: Start cannot be greater than end position.");
        
    MatrixXd m(e1-s1, e2-s2);
    for(int i=0; i<e1-s1; i++){
        for(int j=0; j<e2-s2; j++){
            m(i,j) = matrix(indices[i+s1], j+s2);
        }
    }
    return m;
}

MatrixXd shuffle(const MatrixXd &matrix1, int random_state){
    if(random_state != -1)
        srand(random_state); //seed random number function
    else
        srand(time(0));
    VectorXi indices = VectorXi::LinSpaced(matrix1.rows(), 0, matrix1.rows());
    std::random_shuffle(indices.data(), indices.data() + matrix1.rows());
    MatrixXd res = indices.asPermutation() * matrix1;
    return res;
}

MatrixXd shuffle(const MatrixXd &matrix1, const VectorXi indices, int random_state){
    if(random_state != -1)
        srand(random_state); //seed random number function
    else
        srand(time(0));
    MatrixXd res = indices.asPermutation() * matrix1;
    return res;
}

MatrixXd concatenate(MatrixXd m1, MatrixXd m2, int axis){
    if(axis==0){
        if(m1.cols()!=m2.cols())
            throw std::invalid_argument("concatenate: Inconsistent matrix dimensions.");
        MatrixXd res(m1.cols(), m1.rows()+m2.rows());
        res << m1,
               m2;
        return res;
    }else{
        if(m1.rows()!=m2.rows())
            throw std::invalid_argument("concatenate: Inconsistent matrix dimensions.");
        MatrixXd res(m1.rows(), m1.cols()+m2.cols());
        res << m1, m2;     
        return res;
    }
}
