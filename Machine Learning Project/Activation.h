#pragma once

#include <iostream>
#include <Eigen/Dense>

#include "Layer.h"

using namespace std;

class Activation :
    public Layer
{
public:
    Activation(string name) :Layer(name, false) {};
    ~Activation() {};

    Eigen::MatrixXd getPanOut();
    void setPanOut(Eigen::MatrixXd panOut);
private:
    Eigen::MatrixXd panOut;
};

