#ifndef AUC_H
#define AUC_H

#include <cmath>
#include <climits>
#include <vector>
#include <algorithm>
#include <Eigen/Dense> 

using namespace Eigen;

double calcAUC      (MatrixXd labels, MatrixXd scores, int posclass);
double trapezoidArea(double X1, double X2, double Y1, double Y2);

#endif