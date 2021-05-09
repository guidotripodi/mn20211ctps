#include <algorithm>
#include <numeric>
//#include <chrono>
#include <iostream>
#include "knn.h"
#include <typeinfo>

using namespace std;


KNNClassifier::KNNClassifier(unsigned int n_neighbors)
{
    this->neighbors = n_neighbors;
}

void KNNClassifier::fit(SparseMatrix X, Matrix y)
{
    this->training_samples = X;
    this->training_labels = y;
}


Vector KNNClassifier::predict(SparseMatrix X)
{
    // Creamos vector columna a devolver
    auto ret = Vector(X.rows());

    for (unsigned k = 0; k < X.rows(); ++k)
    {
        ret(k) = this->predict_row(X.row(k)); 
    }

    return ret;
}


// Calculates the distances from every row of the training set
// to the row received as an argument
Vector KNNClassifier::distances_to_row(Vector row)
{ 
    // norm(training - row)
    int rows = this->training_samples.rows();
    auto distances = Vector(rows);
    for(int i = 0; i < rows; ++i)
    {
        distances(i) = (this->training_samples.row(i) - row.transpose()).norm();
    }

    return distances;
}

bool KNNClassifier::predict_row(Vector row)
{
    Vector distances = this->distances_to_row(row);

    // index = np.argsort(dist)
    std::vector<int> closestIndex(distances.size());
    std::iota(closestIndex.begin(), closestIndex.end(), 0);// vector de indices

    //ordeno los indices, usando como orden la distancia respectiva
    std::sort(closestIndex.begin(),
              closestIndex.end(),
              [&distances](size_t i1, size_t i2) {return distances[i1] < distances[i2];});

    // closest = index[0:self.n_neighbors]
    // neighbors = [self.y[i] for i in closest]
    std::vector<bool> closest_neighbors(this->neighbors);
    for(int i = 0; i < this->neighbors; i++)
    {
        int index = closestIndex[i];
        closest_neighbors[i] = this->training_labels(index);
    }

    //count = np.bincount(neighbors)
    //ret = np.argmax(count)
    bool prediction = vote_popular(closest_neighbors);
    return prediction;
}

bool KNNClassifier::vote_popular(std::vector<bool> closest_neighbors)
{ 
    unsigned int positive_reviews = std::count(closest_neighbors.begin(), closest_neighbors.end(), true);
    return (positive_reviews > (closest_neighbors.size()/2));
}
