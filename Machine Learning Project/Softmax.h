#pragma once

#include <iostream>
#include <Eigen/Dense>

#include "Activation.h"

using namespace std;

class Softmax :
    public Activation
{
public:
    Softmax() :Activation("Softmax") {};

    ~Softmax() {};

    // Layer을(를) 통해 상속됨
    virtual Eigen::MatrixXd forward(Eigen::MatrixXd panIn) override;
    virtual Eigen::MatrixXd backward(Eigen::MatrixXd preDiff) override;

    virtual ostream& printConfig(ostream& os) override;
    virtual ostream& printConfig(ostream& os, unsigned int level) override;
};

