#pragma once
#include "Layer.h"
class Sigmoid :
    public Layer
{
public:
    Sigmoid() :Layer("Sigmoid") {};

    virtual Eigen::MatrixXd forward(Eigen::MatrixXd panIn) override;
    virtual Eigen::MatrixXd backward(Eigen::MatrixXd preDiff) override;

    virtual ostream& printConfig(ostream& os) override;
    virtual ostream& printConfig(ostream& os, unsigned int level) override;
private:
    Eigen::MatrixXd panOut;

};

