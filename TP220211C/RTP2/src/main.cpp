//
// Created by pachi on 5/6/19.
//

#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include "catch.hpp"
#include <iostream>
#include <iomanip>
#include "pca.h"
#include "knn.h"
#include "eigen.h"
using namespace std;

TEST_CASE( "Classifying training set returns the same labels", "[knn]" ) {
    KNNClassifier knn(1);
    Matrix y_train(20, 1);
    SparseMatrix x_train(20,20);
    bool act = false;
    for(int i = 0; i < 20; i++)
    {
        act = !act;
        y_train.coeffRef(i,0) = !act;
        for(int j = 0; j<20; j++)
        {
            x_train.coeffRef(i,j) = 10*i * j;
        }
    }

    knn.fit(x_train, y_train);
    Vector y_pred = knn.predict(x_train);
    for(int i = 0; i < 20; i++)
    {
        REQUIRE(y_pred.coeff(i, 0) == y_train.coeff(i, 0));
    }
}

TEST_CASE( "Transforming with PCA", "[pca]" ) {
    PCA pca(5);
    Matrix y_train(20, 1);
    SparseMatrix x_train(20,19);
    bool act = false;
    for(int i = 0; i < 20; i++)
    {
        act = !act;
        y_train.coeffRef(i,0) = !act;
        for(int j = 0; j<19; j++)
        {
            x_train.coeffRef(i,j) = (double) 10*(i+2) + (j+1/33);
        }
    }
    cout << "Matrix built\n";

    pca.fit(x_train);
    Eigen::MatrixXd train_new = pca.transform(x_train);
}
