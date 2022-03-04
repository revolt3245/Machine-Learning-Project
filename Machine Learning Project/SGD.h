#pragma once
#include "Optimizer.h"
class SGD :
    public Optimizer
{
public:
    SGD(double learningRate, double regL2)
        :Optimizer(learningRate, regL2) {};
    ~SGD() {};

    virtual void step(Learnable* obj) override;
private:
};

