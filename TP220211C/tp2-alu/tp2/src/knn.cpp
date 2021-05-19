#include <algorithm>
//#include <chrono>
#include <iostream>
#include "knn.h"
#include "pca.h"

using namespace std;

KNNClassifier::KNNClassifier(unsigned int n_neighbors) {
    this->n_neighbors = n_neighbors;
}

void KNNClassifier::fit(Matrix X, Matrix y) {
    this->x_train = X;
    this->y_train = y;
}

Vector KNNClassifier::predict(Matrix X) {
    // Creamos vector columna a devolver
    auto ret = Vector(X.rows());

    for (unsigned k = 0; k < X.rows(); ++k) {
        ret(k) = calculate_KNN(X.row(k));
    }
    return ret;
}

bool compareFirstCoordinate(pair<double, unsigned int> pair1, pair<double, unsigned int> pair2) {
    return (pair1.first < pair2.first);
}

unsigned int KNNClassifier::get_mode(std::vector<pair<double, unsigned int>> distances) {
    std::vector<int> histogram(10, 0);
    unsigned int max = min(n_neighbors, (unsigned int) distances.size());
    for (unsigned int i = 0; i < max; ++i)
        ++histogram[distances[i].second];
    return (std::max_element(histogram.begin(), histogram.end())  - histogram.begin());
}

unsigned int KNNClassifier::calculate_KNN(Vector vector_to_predict) {
    std::vector<pair<double, unsigned int>> distances(this->x_train.rows());
    for (unsigned i = 0; i < x_train.rows(); i++) {
        double distance =  (vector_to_predict.transpose() - this->x_train.row(i)).norm();
        auto digit_class = (unsigned int) this->y_train.coeff(i,0);
        distances[i] = (make_pair(distance, digit_class));
    }

    std::sort(distances.begin(), distances.end(), compareFirstCoordinate);

    return get_mode(distances);
}