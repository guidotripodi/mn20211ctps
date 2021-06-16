#include <algorithm>
//#include <chrono>
#include <pybind11/pybind11.h>
#include <iostream>
#include <exception>
#include "linear_regression.h"

using namespace std;
namespace py=pybind11;

LinearRegression::LinearRegression()
{
}

/**
 * Lee nuestro set de entrenamiento y entrena el modelo.
 * Como es regresion lineal, hacemos CML asumiendo que las funciones que
 * conforman la matriz A son f(xi) siendo xi una feature, ademas de un termino
 * independiente.
 *
 * @param Matrix X : n muestras x m features
 * @param Matrix y : n muestras
 */
void LinearRegression::fit(Matrix X, Matrix y)
{
    // Creamos matriz de 1s y reemplazamos las primeras m columnas con X
    // TODO: hacer algo mas eficiente
    auto A = Matrix(X.rows(), X.cols()+1);// nxm+1
    auto ones = MatrixXd::Ones(X.rows(), 1);
    A << X, ones;

    // Ecuaciones normales
    auto AtA = A.transpose() * A;// m+1xn * nxm+1 = m+1xm+1
    auto Atb = A.transpose() * y;// m+1xn * nx1 = m+1x1
    this->model = AtA.lu().solve(Atb);//m+1x1
}

/**
 * Dado un set de test, para cada uno predice el valor de la
 * variable dependiente
 *
 * @param Matrix X : l muestras x m features
 * @returns Matrix : l predicciones
 **/
Matrix LinearRegression::predict(Matrix X)
{
    //Agregamos una feature "independiente"
    auto A = Matrix(X.rows(), X.cols()+1);// nxm+1
    auto ones = MatrixXd::Ones(X.rows(), 1);
    A << X,ones;

    return A * this->model;// lxm+1 * m+1x1 = lx1
}
