#include <algorithm>
#include <chrono>
#include <iostream>
#include "eigen.h"
#include <cmath>
using namespace std;


pair<double, Vector> power_iteration(const Matrix& X, unsigned num_iter, double eps)
{
    Vector b = Vector::Random(X.cols());
    Vector old = Vector::Zero(X.cols());
    double eigenvalue;
    b.normalize();
    unsigned int iter=0;
    double dif = abs((old-b).norm());
    
    while(dif > eps && iter < num_iter){
        old = b;
        b = X*b;
        b.normalize();
        dif = abs((old-b).norm());
        iter++;
    }

    eigenvalue = (X*b).norm() ;
    return make_pair(eigenvalue, b / b.norm());
}

pair<Vector, Matrix> get_first_eigenvalues(const Matrix& X, unsigned num, unsigned num_iter, double epsilon)
{
    Matrix A(X);
    Vector eigvalues(num);
    Matrix eigvectors(num, A.cols());

    for(unsigned int i = 0; i < num; i++){
        pair<double, Vector> av = power_iteration(A, num_iter, epsilon);
        double lambda = av.first;
        Vector v = av.second;
        eigvectors.row(i) = v;
        eigvalues(i) = lambda;
        A = A - (lambda*v*v.transpose());
    }

    return make_pair(eigvalues, eigvectors);
}
