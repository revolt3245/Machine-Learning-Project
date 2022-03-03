#include "Linear.h"

Eigen::MatrixXd Linear::forward(Eigen::MatrixXd panIn)
{
    auto batchSize = panIn.rows();
    auto panOut = panIn * this->weight + Eigen::MatrixXd::Ones(batchSize, 1) * this->bias;

    this->weightGrad = panIn.transpose();
    return panOut;
}

ostream& Linear::printConfig(ostream& os)
{
    os << this->getName() << "(" << this->input << ", " << this->output << ")";

    return os;
}

Eigen::MatrixXd Linear::backward(Eigen::MatrixXd preDiff)
{
    auto batchSize = preDiff.rows();
    this->biasGrad = Eigen::MatrixXd::Ones(1, batchSize) * preDiff;

    this->weightGrad *= preDiff;

    return preDiff * this->weight.transpose();
}

void Linear::step(double learningRate)
{
    this->weight -= learningRate * (1e-5 * this->weight + this->weightGrad);
    this->bias -= learningRate * this->biasGrad;
}

Eigen::MatrixXd Linear::getWeight()
{
    return this->weight;
}

Eigen::MatrixXd Linear::getBias()
{
    return this->bias;
}
