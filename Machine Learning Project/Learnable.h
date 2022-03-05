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

    //vector<Eigen::MatrixXd*> getParam();
    //vector<Eigen::MatrixXd*> getGrad();
    //vector<double> getL2Regularization();

    //void addParam(Eigen::MatrixXd* param);
    //void addParam(Eigen::MatrixXd* param, Eigen::MatrixXd* grad);

    vector<Parameter*> getParam();
    void addParam(Parameter* param);
    void addParam(vector<Parameter*> params);

    //void addParam(vector<Eigen::MatrixXd*> params);
    //void addParam(vector<Eigen::MatrixXd*> params, vector<Eigen::MatrixXd*> grads);;
    //void setL2Regularization(vector<double> regL2);
    //void setL2Regularization(size_t idx, double regL2);
private:
    //vector<Eigen::MatrixXd*> param;
    //vector<Eigen::MatrixXd*> grad;
    //vector<double> regL2;

    vector<Parameter*> param;
};

