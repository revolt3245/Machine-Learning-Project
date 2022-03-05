#pragma once

#include <iostream>
#include <Eigen/Dense>

#include "Layer.h"

using namespace std;

class ReLU :
    public Layer
{
public:
    virtual Eigen::MatrixXd forward(Eigen::MatrixXd panIn) override;
    virtual Eigen::MatrixXd backward(Eigen::MatrixXd preDiff) override;
    virtual ostream& printConfig(ostream& os) override;
    virtual ostream& printConfig(ostream& os, unsigned int level) override;
private:
    Eigen::MatrixXd panOut;
};

