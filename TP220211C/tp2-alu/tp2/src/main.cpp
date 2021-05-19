#include <iostream>
#include <sys/time.h>
#include "pca.h"
#include "eigen.h"
#include "CSVEigenConverter.h"
#include "knn.h"
#include "Matrix_printer.h"

typedef struct args_t {
    std::string method;
    std::string training_set;
    std::string testing_set;
    std::string classif;
    std::string mode;
    std::string base_change_matrix_path;
    std::string X_matrix_path;
    std::string Y_matrix_path;
} Arguments; 

// neighbors for finding the mode in KNN
unsigned int N_NEIGHBORS = 5;
// components for PCA
unsigned int N_COMPONENTS = 20;
// number of iterations to find eigenvalues and eigenvectors in power iteration
unsigned int N_ITERATIONS = 2000;
// epsilon for power iteration
double EPSILON = 1e-10;
// percentage of train cases used to train the model.
unsigned int PERCENTAGE_OF_TRAIN_CASES = 1;
const char *methods[2] = {"KNN", "KNN+PCA"};
const char *modes[3] = {"TRAIN", "PREDICT", "TRAIN+PREDICT"};

void show_stats(double time, std::string method);
void save_to_file(std::string, Vector matrix);
timeval start_time();
double end_time(timeval start);
Arguments parse_arguments(int argc, char **pString);


int main(int argc, char **argv) {

    timeval start = start_time();

    Arguments args = parse_arguments(argc, argv);

    CSVEigenConverter<MatrixXd> converter = CSVEigenConverter<MatrixXd>();

    if (args.mode == modes[0]) {
        Matrix temp = converter.load_csv(args.training_set, true);
        Matrix X = temp.block(0, 1, temp.rows(), temp.cols() - 1);
        Matrix Y = temp.block(0, 0, temp.rows(), 1);

        if (args.method == methods[1]) {
            PCA pca = PCA(N_COMPONENTS, N_ITERATIONS, EPSILON);
            pca.fit(X);
            X = pca.transform(X);
            converter.writeToCSVfile(
                args.base_change_matrix_path,
                pca.get_base_change_matrix()
            );
        }

        converter.writeToCSVfile(args.X_matrix_path, X);
        converter.writeToCSVfile(args.Y_matrix_path, Y);

    } else if (args.mode == modes[1]) {
        Matrix X = converter.load_csv(args.X_matrix_path, true);
        Matrix Y = converter.load_csv(args.Y_matrix_path, true);
        Matrix X_test_temp = converter.load_csv(args.testing_set, true);
        Matrix X_test = X_test_temp.block(0, 1, X_test_temp.rows(), X_test_temp.cols() - 1);
        Matrix base_change_matrix = converter.load_csv(args.base_change_matrix_path, true);

        if (args.method == methods[1]) {
            X_test = X_test * base_change_matrix;
        }

        KNNClassifier knn = KNNClassifier(N_NEIGHBORS);
        knn.fit(X, Y);
        Vector y_pred = knn.predict(X_test);
        save_to_file(args.classif, y_pred);

    } else if (args.mode == modes[2]) {
        Matrix temp = converter.load_csv(args.training_set, true);
        KNNClassifier knn = KNNClassifier(N_NEIGHBORS);
        Matrix X = temp.block(0, 1, temp.rows(), temp.cols() - 1);
        Matrix Y = temp.block(0, 0, temp.rows(), 1);
        Matrix X_test = converter.load_csv(args.testing_set, true);

        if (args.method == methods[1]) {
            PCA pca = PCA(N_COMPONENTS, N_ITERATIONS, EPSILON);
            pca.fit(X);
            X = pca.transform(X);
            X_test = pca.transform(X_test);
        }

        knn.fit(X, Y);
        Vector y_pred = knn.predict(X_test);
        save_to_file(args.classif, y_pred);
    }

    double time = end_time(start);
    show_stats(time, args.method);

    return 0;
}

void showUsage(){
    const std::string usage = R"USAGE(
usage: ./tp2 -m METHOD -i TRAIN -q TEST -o OUTPUT [exec_mode base_change_matrix_path X_matrix_path Y_matrix_path]

Digit recognition using KNN and PCA.

Arguments:
    -m METHOD    Specify 0 for knn or 1 for knn and pca
    -i TRAIN     Path to the training set.
    -q TEST      Path to the test set.
    -o OUTPUT    Path where the classification results will be written to.
    exec_mode = 0 : train
                1 : predict
                2 : train+predict
    base_change_matrix_path : ex. change_base_matrix.csv.
    X_matrix_path : ex. X.csv.
    Y_matrix_path : ex. Y.csv.
)USAGE";
    std::cout << usage << "\n"; 
}

Arguments parse_arguments(int argc, char **argv) {
    int required_amount = 9; // 2*4 (flag and values) + 1 (program name)
    if (argc < required_amount) {
        showUsage();
        exit(1);
    }
    
    Arguments arg;
    arg.mode = modes[2];
    arg.base_change_matrix_path = "data/base_change.csv";
    arg.X_matrix_path = "data/x_matrix.csv";
    arg.Y_matrix_path = "data/y_matrix.csv";

    for (int i = 1; i < required_amount; i+=2) {
        std::string argument = argv[i];
        if (argument == "-m") {
            int methodId = strtol(argv[i+1], NULL, 10);
            if (methodId < 0 || methodId > 1) {
                std::cout << "Unknown method: " << methodId << "\n";
                showUsage();
                exit(1);
            }
            arg.method = methods[methodId];
        } else if (argument == "-i") {
            arg.training_set = argv[i+1];
        } else if (argument == "-q") {
            arg.testing_set = argv[i+1];
        } else if (argument == "-o") {
            arg.classif = argv[i+1];
        } else {
            std::cout << "Unknown option: " << argument << "\n";
            showUsage();
            exit(1);
        }
    }

    int mode_index = required_amount;
    int optional_amount = 11;
    if(argc > required_amount){
        if(argc < optional_amount){
            showUsage();
            exit(1);
        }

        arg.mode = argv[mode_index];
        if (strcmp(argv[mode_index], "0") == 0) {
            arg.mode = modes[0];
        } else if (strcmp(argv[mode_index], "1") == 0) {
            arg.mode = modes[1];
        } else if (strcmp(argv[mode_index], "2") == 0) {
            arg.mode = modes[2];
        }

        // Paths where to save the matrix that are created in training (in train mode - 0),
        // or where to load them from (in predict mode - 1).
        // Can be whatever for mode 2 (train+predict at once - 2)
        arg.base_change_matrix_path = argv[mode_index + 1];
        arg.X_matrix_path = argv[mode_index + 2];
        arg.Y_matrix_path = argv[mode_index + 3];
    }

    return arg;
}

timeval start_time() {
    timeval start;
    gettimeofday(&start, NULL);
    return start;
}

double end_time(timeval start) {
    timeval end;
    gettimeofday(&end, NULL);
    long elapsed_seconds = end.tv_sec - start.tv_sec; /* diff between seconds counter */
    long elapsed_useconds = end.tv_usec - start.tv_usec; /* diff between microseconds counter */
    return ((elapsed_seconds) * 1000 + elapsed_useconds / 1000.0) + 0.5;
}


void save_to_file(std::string name, Vector content) {
    //preparo archivo salida para escritura
    std::ofstream archivoDeSalida;
    archivoDeSalida.setf(std::ios::fixed, std::ios::floatfield); // tipo salida
    archivoDeSalida.precision(15); // cant decimales
    archivoDeSalida.open(name);
    for (unsigned int i = 0; i < content.rows(); i++) {
        archivoDeSalida << (unsigned int) content[i] << std::endl;
    }
    archivoDeSalida.close();

}

void show_stats(double time, std::string method) {
    unsigned int percentage = (unsigned int) PERCENTAGE_OF_TRAIN_CASES * 100;
    std::cout << "------------------------°------------------------" << std::endl;
    std::cout << "Finished in: " << (unsigned int) time / 1000 << " seconds." << std::endl;
    std::cout << "Method used in training: " << method << std::endl;
    std::cout << "Number of neighbors used in KNN: " << N_NEIGHBORS << std::endl;
    std::cout << "Number of components used in PCA: " << N_COMPONENTS << std::endl;
    std::cout << "Number of iterations used in power iteration: " << N_ITERATIONS << std::endl;
    std::cout << "Epsilon used in power iteration: " << EPSILON << std::endl;
    std::cout << "Percentage of train data used to fit model : " << percentage << "%" << std::endl;
    std::cout << "------------------------°------------------------" << std::endl;
}
