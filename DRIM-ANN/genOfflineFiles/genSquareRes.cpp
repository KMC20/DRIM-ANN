/*
Author: KMC20
Date: 2024/3/25
Function: Generate the square results offline for DRIM-ANN.
Usage:
 > g++ genSquareRes.cpp -std=c++11 -o genSquareRes
 > ./genSquareRes ../DRIM-ANN/offlineFiles/squareResUint16
 > rm genSquareRes
*/

#include <fstream>
#include <iostream>
#include <vector>
#include <string>

#define NR_JOB_PARALLEL 64

using UINT64 = unsigned long long;
using ELEMTYPE = unsigned char;
using ELEMSUMTYPE = unsigned short;

int main(int argc, char **argv)
{
    std::string squareResFileName = "squareResUint8";
    if (argc > 2) {
        std::cerr << "The amount of input parameters of this program cannot become more than one! Exit now!" << std::endl;
        return -2;
    } else if (argc > 1) {
        squareResFileName = argv[1];
    }
    UINT64 upperBound = 1 << (sizeof(ELEMTYPE) << 3);
    std::vector<ELEMSUMTYPE> res(upperBound);
    #pragma omp parallel for num_threads(NR_JOB_PARALLEL)
    for (ELEMSUMTYPE elem = 0; elem < upperBound; ++elem)
        res[elem] = elem * elem;
    std::fstream file;
    file.open(squareResFileName, std::ios::out | std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open the output square result file: " << squareResFileName << "! Exit now!" << std::endl;
        file.close();
        return -1;
    }
    file.write(reinterpret_cast<char *>(res.data()), upperBound * sizeof(ELEMSUMTYPE));
    file.close();
    return 0;
}