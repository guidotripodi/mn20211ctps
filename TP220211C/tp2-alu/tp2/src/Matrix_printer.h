//
// Created by jazzer on 6/6/20.
//

#ifndef TP2_MATRIX_PRINTER_H
#define TP2_MATRIX_PRINTER_H


#include <string>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <iostream>

template <typename M>
class Matrix_printer {
public:
    static void print_matrix(M &A, std::string name, unsigned int limit = 0) {
        std::cout << "\n--------------" << name << "----------------\n" << std::endl;
        if (limit > 0) {
            for (unsigned int i = 0; i < limit; i++) {
                    std::cout << A.row(i) << std::endl;
            }
        } else {
            std::cout << A << std::endl;
        }
        std::cout << "\n------------------------------------------\n" << std::endl;
    }
};


#endif //TP2_MATRIX_PRINTER_H
