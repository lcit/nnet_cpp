#ifndef EIGEN_PLUS_H
#define EIGEN_PLUS_H

#include <Eigen/Dense>

using namespace Eigen;

MatrixXd slice(const MatrixXd &matrix, const int s1, const int e1);
MatrixXd slice(const MatrixXd &matrix, const VectorXi indices, const int s1, const int e1);
MatrixXd slice(const MatrixXd &matrix, const int s1, const int e1, const int s2, const int e2);
MatrixXd slice(const MatrixXd &matrix, const VectorXi indices, const int s1, const int e1, const int s2, const int e2);

MatrixXd shuffle(const MatrixXd &matrix1, int random_state);
MatrixXd shuffle(const MatrixXd &matrix1, const VectorXi indices, int random_state);

MatrixXd concatenate(MatrixXd m1, MatrixXd m2, int axis);

#endif