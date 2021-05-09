#pragma once

#include "types.h"


class KNNClassifier {
public:
    KNNClassifier(unsigned int n_neighbors);

    void fit(SparseMatrix X, Matrix y);

    Vector predict(SparseMatrix X);
private:
    Vector distances_to_row(Vector row);
    bool vote_popular(std::vector<bool> closest_neighbors);
    bool predict_row(Vector row);
    int neighbors;
    Matrix training_samples;
    Matrix training_labels;
};
