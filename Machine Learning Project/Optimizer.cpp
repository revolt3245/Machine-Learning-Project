#include "Optimizer.h"

double Optimizer::getLearningRate()
{
    return this->learningRate;
}

double Optimizer::getL2Regularization()
{
    return this->regL2;
}

void Optimizer::setLearningRate(double learningRate)
{
    this->learningRate = learningRate;
}

void Optimizer::setL2Regularization(double regL2)
{
    this->regL2 = regL2;
}

vector<Parameter*> Optimizer::getLearnableParam(Learnable* obj)
{
    return obj->param;
}
