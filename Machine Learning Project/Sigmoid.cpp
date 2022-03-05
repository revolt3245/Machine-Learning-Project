#include "Sigmoid.h"

Eigen::MatrixXd Sigmoid::forward(Eigen::MatrixXd panIn)
{
    this->setPanOut(panIn.array().logistic().matrix());
    return this->getPanOut();
}

Eigen::MatrixXd Sigmoid::backward(Eigen::MatrixXd preDiff)
{
    return (this->getPanOut().array() * (1 - this->getPanOut().array()) * preDiff.array()).matrix();
}

ostream& Sigmoid::printConfig(ostream& os)
{
    os << this->getName();

    return os;
}

ostream& Sigmoid::printConfig(ostream& os, unsigned int level)
{
    for (auto i = 0; i < level; i++) {
        os << "    ";
    }
    os << this->getName();

    return os;
}
