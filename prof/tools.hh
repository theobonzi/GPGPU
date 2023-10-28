#ifndef TOOLS_HH
#define TOOLS_HH

#include <iostream>
#include <vector>

namespace tifo{
    // Multiplication de matrix
    float mat_mul(std::vector<float> mat1, std::vector<float> mat2){
        if (mat1.size() != mat2.size()){
            std::cout << "error Mat mult\n Mat1: " << mat1.size()
                << " and Mat2: " << mat2.size() << "\n";
        }
        float res = 0;
        for (int i = 0; i < mat1.size(); i++){
            res += mat1[i] * mat2[i];
        }
        return res;
    }
    float sum_kernel(std::vector<float> mat){
        float res = 0;
        for (int i = 0; i < mat.size(); i+=1){
            res += mat[i];
        }
        if (res < 0){
            // std::cout << "ZERO\n";
            return -res;
        }
        return res;
    }
    std::ostream& operator<<(std::ostream& os, std::vector<float> mat){
        os << "{ ";
        for (int i = 0; i < mat.size(); i+=1){
            os << mat[i] << ",";
        }
        os << "}";
        return os;
    }
};

#endif
