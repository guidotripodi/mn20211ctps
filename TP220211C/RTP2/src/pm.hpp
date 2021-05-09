#include "types.h"
Vector crearVectorInicial(int dim);
double encontrarAutovalor(Vector autovector, Matrix m);
Vector metodoDeLasPotencias(Matrix covarianza);

Vector metodoDeLasPotencias(Matrix covarianza) {
    int dimension = covarianza.cols();
    Vector vectorInicial = crearVectorInicial(dimension);
    double normaVieja = 0.0;
    double norma = 1.0;

    while(abs( normaVieja - norma) > 0.00000000000000001) {
        normaVieja = vectorInicial.norm();
        Vector nuevoVector = covarianza * vectorInicial;
        norma = nuevoVector.norm();
        vectorInicial = nuevoVector * (1 / (double)norma);
    }
    return vectorInicial;
}

Vector crearVectorInicial(int dim) {
    Vector vectorInicial(dim);
    for (int w = 0; w < dim; w++) {
        double random = rand() % 100 + 1.0;
        vectorInicial(w) = (double)random;
    }
    return vectorInicial;
}

double encontrarAutovalor(Vector autovector, Matrix m) {
    Vector aux = m * autovector;
    double lamda = aux.norm();
    return lamda;
}

