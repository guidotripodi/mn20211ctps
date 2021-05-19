#pragma once
#include "types.h"

class PCA {
public:
    PCA(unsigned int n_components, unsigned num_iter, double epsilon);

    void fit(Matrix X);

    Eigen::MatrixXd transform(Matrix X);

    Matrix get_base_change_matrix();

private:
    unsigned int n_components;
    unsigned num_iter;
    double epsilon;

    Matrix calculate_covariance(Matrix matrix);

    Matrix base_change_matrix;
};
