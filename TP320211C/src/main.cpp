#include <iostream>
#include "linear_regression.h"
#include "eigen.h"
#include "types.h"

int main(int argc, char **argv)
{

  std::cout << "Hola mundo!" << std::endl;

  auto X = Matrix::Zero(1, 1);
  auto y = Matrix::Zero(1, 1);
  auto testSet = Matrix::Zero(1, 1);

  regression = LinearRegression();
  regression.fit(X, y);
  regression.predict(testSet);
  return 0;
}
