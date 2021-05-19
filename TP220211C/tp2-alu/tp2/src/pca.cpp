#include <iostream>
#include "pca.h"
#include "eigen.h"
#include "Matrix_printer.h"

using namespace std;

PCA::PCA(unsigned int n_components, unsigned num_iter, double epsilon) {
    this->n_components = n_components;
    this->num_iter = num_iter;
    this->epsilon = epsilon;
}

void PCA::fit(Matrix X) {
    Matrix covariance = calculate_covariance(X);

    auto eigenpairs = get_first_eigenvalues(
        covariance,
        this->n_components,
        this->num_iter,
        this->epsilon
    );
    this->base_change_matrix = eigenpairs.second;
}

MatrixXd PCA::transform(Matrix X) {
    return X * base_change_matrix;
}

Matrix PCA::calculate_covariance(Matrix matrix) {
    auto centered = matrix.rowwise() - matrix.colwise().mean();
    return (centered.transpose() * centered) / double(matrix.rows() - 1);
}

Matrix PCA::get_base_change_matrix(){
    return base_change_matrix;
}
