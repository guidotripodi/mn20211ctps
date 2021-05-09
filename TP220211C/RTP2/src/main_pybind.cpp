//
// Created by pachi on 5/6/19.
// Ejemplo basado en la pregunta de stackoverflow:
// https://stackoverflow.com/questions/47762543/segfault-when-using-pybind11-wrappers-in-c

#include <iostream>
#include "pca.h"
#include "knn.h"
#include "eigen.h"
#include "types.h"

#include <pybind11/embed.h>
#include <pybind11/eigen.h>

namespace py = pybind11;

int main(int argc, char** argv){
    KNNClassifier knn(1);
    Matrix y_train(20, 1);
    SparseMatrix x_train(20,20);
    bool act = false;
    for(int i = 0; i < 20; i++)
    {
        act = !act;
        y_train.coeffRef(i,0) = act;
        for(int j = 0; j<20; j++)
        {
            x_train.coeffRef(i,j) = 10*i + j;
        }
    }

    knn.fit(x_train, y_train);
    Vector y_pred = knn.predict(x_train);

    py::scoped_interpreter guard{};
    py::print("Training labels:");
    py::print(y_train);
    py::print("Predictions:");
    py::print(y_pred);

    return 0;
}
