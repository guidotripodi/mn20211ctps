//
// Created by jazzer on 6/6/20.
//

#include <string>
#include <fstream>

#ifndef TP2_CSVTOEIGEN_H
#define TP2_CSVTOEIGEN_H

const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision,
                                       Eigen::DontAlignCols,
                                       ", ",
                                       "\n");

template<typename M>
class CSVEigenConverter {
public:

    void writeToCSVfile(std::string name, M matrix)
    {
        std::ofstream file(name.c_str());
        file << matrix.format(CSVFormat);
    }

    M load_csv(const std::string &path, const bool &skip_headers = false) {
        std::ifstream indata;
        indata.open(path);
        std::string line;
        std::vector<double> values;
        uint rows = 0;

        if (skip_headers) {
            std::getline(indata, line);
            std::stringstream lineStream(line);
            std::string cell;
            while (std::getline(lineStream, cell, ',')) {
            }
        }

        while (std::getline(indata, line)) {
            std::stringstream lineStream(line);
            std::string cell;
            while (std::getline(lineStream, cell, ',')) {
                values.push_back(std::stod(cell));
            }
            ++rows;
        }
        return Eigen::Map<const Eigen::Matrix<typename M::Scalar, M::RowsAtCompileTime, M::ColsAtCompileTime, Eigen::RowMajor>>(
                values.data(), rows, values.size() / rows);
    }
};

#endif //TP2_CSVTOEIGEN_H
