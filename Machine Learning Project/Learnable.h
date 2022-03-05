#pragma once

#include <iostream>
#include <vector>
#include <Eigen/Dense>

#include "Layer.h"

struct Parameter {
    Eigen::MatrixXd value;
    Eigen::MatrixXd grad;

    double regL2;
};

class Learnable :
    public Layer
{
public:
    friend class Optimizer;
    Learnable(string name) 
        :Layer(name, true), param(0) {};
    Learnable(string name, size_t n) 
        :Layer(name, true), param(n) {
        for (int i = 0; i < n; i++) {
            param[i] = new Parameter;
        }
    };

    ~Learnable() {
        for (auto& p : param) {
            if(p != nullptr) delete p;
        }
    };

    vector<Parameter*> getParam();
    void addParam(Parameter* param);
    void addParam(vector<Parameter*> params);
    
private:
    vector<Parameter*> param;
};

