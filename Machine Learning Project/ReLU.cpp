#include "ReLU.h"

Eigen::MatrixXd ReLU::forward(Eigen::MatrixXd panIn)
{
    this->setPanOut((panIn.array() >= 0).cast<double>().matrix());
    Eigen::MatrixXd Out = (this->getPanOut().array() * panIn.array()).matrix();
    return Out;
}

Eigen::MatrixXd ReLU::backward(Eigen::MatrixXd preDiff)
{
    return (this->getPanOut().array() * preDiff.array()).matrix();
}

ostream& ReLU::printConfig(ostream& os)
{
    os << this->getName();
    return os;
}

ostream& ReLU::printConfig(ostream& os, unsigned int level)
{
    for (auto i = 0; i < level; i++) {
        os << "\t";
    }
    os << this->getName();

    return os;
}
