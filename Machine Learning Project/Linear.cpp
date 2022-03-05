#include "Linear.h"

Eigen::MatrixXd Linear::forward(Eigen::MatrixXd panIn)
{
    auto batchSize = panIn.rows();
    auto Param = this->getParam();

    auto panOut = panIn * Param[0]->value + Eigen::MatrixXd::Ones(batchSize, 1) * Param[1]->value;

    Param[0]->grad = panIn.transpose();
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
    auto Param = this->getParam();
    
    Param[0]->grad *= preDiff;
    Param[1]->grad = Eigen::MatrixXd::Ones(1, batchSize) * preDiff;
    return preDiff * Param[0]->value.transpose();
}

Eigen::MatrixXd Linear::getWeight()
{
    return this->getParam()[0]->value;
}

Eigen::MatrixXd Linear::getBias()
{
    return this->getParam()[1]->value;
}
