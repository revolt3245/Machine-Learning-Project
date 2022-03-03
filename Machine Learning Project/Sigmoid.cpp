#include "Sigmoid.h"

Eigen::MatrixXd Sigmoid::forward(Eigen::MatrixXd panIn)
{
    return this->panOut;
}

Eigen::MatrixXd Sigmoid::backward(Eigen::MatrixXd preDiff)
{
    return (panOut.array() * (1 - panOut.array())).matrix();
}

ostream& Sigmoid::printConfig(ostream& os)
{
    os << this->getName();

    return os;
}
