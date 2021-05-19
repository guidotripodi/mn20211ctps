#include <algorithm>
#include <chrono>
#include <iostream>
#include "eigen.h"

using namespace std;


pair<double, Vector> power_iteration(const Matrix& X, unsigned num_iter, double eps)
{
    Vector b = Vector::Random(X.cols());
    b = b / b.norm();
    Vector prev;

    for (unsigned int i = 0; i < num_iter; i++) {
        prev = b;
        b = X * b;
        b = b / b.norm();
        double cos = b.dot(prev);
        if (cos > 1 - eps) {
            i = num_iter + 1;
        }
    }
    double eigenvalue = b.dot(X * b);
    return make_pair(eigenvalue, b);
}

pair<Vector, Matrix> get_first_eigenvalues(const Matrix& X, unsigned num, unsigned num_iter, double epsilon)
{
    Matrix A(X);
    Vector eigvalues(num);
    Matrix eigvectors(A.rows(), num);

    for (unsigned i = 0; i < num; i++) {
        pair<double, Vector> eigenpair = power_iteration(A, num_iter, epsilon);
        double eigenvalue = eigenpair.first;
        Vector eigenvector = eigenpair.second;
        eigvalues(i) = eigenvalue;
        eigvectors.col(i) = eigenvector;
        A = A - eigenvalue * eigenvector * eigenvector.transpose();
    }

    return make_pair(eigvalues, eigvectors);
}
