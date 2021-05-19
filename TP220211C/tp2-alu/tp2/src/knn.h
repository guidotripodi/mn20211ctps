#pragma once
#include <algorithm>
//#include <chrono>
#include <iostream>
#include "types.h"
#include "pca.h"


class KNNClassifier {
public:
    KNNClassifier(unsigned int n_neighbors);

    void fit(Matrix X, Matrix y);

    Vector predict(Matrix X);
private:
    unsigned int n_neighbors;
    Vector y_train;
    MatrixXd x_train;

    unsigned int calculate_KNN(Vector type);

    unsigned int get_mode(std::vector<std::pair<double, unsigned int>> distances);
};
