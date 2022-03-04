#pragma once

#include <iostream>
#include <vector>
#include <Eigen/Dense>

#include "Layer.h"

class Learnable :
    public Layer
{
public:
    Learnable(string name) 
        :Layer(name), param(0), grad(0) {};
    Learnable(string name, size_t n) 
        :Layer(name), param(n), grad(n) {
        for (int i = 0; i < n; i++) {
            param[i] = new Eigen::MatrixXd;
            grad[i] = new Eigen::MatrixXd;
        }
    };

    ~Learnable() {
        for (auto& p : param) {
            delete p;
        }
        for (auto& g : grad) {
            delete g;
        }
    };
protected:
    vector<Eigen::MatrixXd*> getParam();
    vector<Eigen::MatrixXd*> getGrad();

    void addParam(Eigen::MatrixXd* param);
    void addParam(vector<Eigen::MatrixXd*> params);
private:
    vector<Eigen::MatrixXd*> param;
    vector<Eigen::MatrixXd*> grad;
};

