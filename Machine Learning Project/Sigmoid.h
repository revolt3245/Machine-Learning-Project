#pragma once
#include "Layer.h"
class Sigmoid :
    public Layer
{
public:
    Sigmoid() :Layer("Sigmoid") {};

    virtual Eigen::MatrixXd forward(Eigen::MatrixXd panIn) override;
    virtual Eigen::MatrixXd backward(Eigen::MatrixXd preDiff) override;
protected:
    virtual ostream& printConfig(ostream& os) override;
private:
    Eigen::MatrixXd panOut;
};

