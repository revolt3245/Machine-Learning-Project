#include "Sigmoid.h"

Eigen::MatrixXd Sigmoid::forward(Eigen::MatrixXd panIn)
{
    this->panOut = panIn.array().logistic().matrix();
    return this->panOut;
}

Eigen::MatrixXd Sigmoid::backward(Eigen::MatrixXd preDiff)
{
    return (panOut.array() * (1 - panOut.array()) * preDiff.array()).matrix();
}

ostream& Sigmoid::printConfig(ostream& os)
{
    os << this->getName();

    return os;
}

ostream& Sigmoid::printConfig(ostream& os, unsigned int level)
{
    for (auto i = 0; i < level; i++) {
        os << "\t";
    }
    os << this->getName();

    return os;
}
