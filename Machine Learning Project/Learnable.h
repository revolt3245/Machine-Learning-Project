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
        :Layer(name, true), param(n) {};

    ~Learnable() {};

    vector<Parameter*> getParam();
    void addParam(Parameter* param);
    void addParam(vector<Parameter*> params);

    void allocParam();
    void deleteParam();
private:
    vector<Parameter*> param;
};

