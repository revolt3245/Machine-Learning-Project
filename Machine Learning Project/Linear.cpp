#include "Linear.h"

Eigen::MatrixXd Linear::forward(Eigen::MatrixXd panIn)
{
    auto batchSize = panIn.rows();
    auto Param = this->getParam();
    auto Grad = this->getGrad();
    auto panOut = panIn * (*Param[0]) + Eigen::MatrixXd::Ones(batchSize, 1) * (*Param[1]);

    (*Grad[0]) = panIn.transpose();
    return panOut;
}

ostream& Linear::printConfig(ostream& os)
{
    os << this->getName() << "(" << this->input << ", " << this->output << ")";

    return os;
}

ostream& Linear::printConfig(ostream& os, unsigned int level)
{
    for (auto i = 0; i < level; i++) {
        os << "\t";
    }
    os << this->getName() << "(" << this->input << ", " << this->output << ")";

    return os;
}

Eigen::MatrixXd Linear::backward(Eigen::MatrixXd preDiff)
{
    auto batchSize = preDiff.rows();
    auto Grad = this->getGrad();
    auto Param = this->getParam();
    (*Grad[1]) = Eigen::MatrixXd::Ones(1, batchSize) * preDiff;

    (*Grad[0]) *= preDiff;

    return preDiff * (*Param[0]).transpose();
}

void Linear::step(double learningRate)
{
    auto Param = this->getParam();
    auto Grad = this->getGrad();
    (*Param[0]) -= learningRate * (1e-5 * (*Param[0]) + (*Grad[0]));
    (*Param[1]) -= learningRate * (*Grad[1]);
}

Eigen::MatrixXd Linear::getWeight()
{
    return *(this->getParam()[0]);
}

Eigen::MatrixXd Linear::getBias()
{
    return *(this->getParam()[1]);
}
