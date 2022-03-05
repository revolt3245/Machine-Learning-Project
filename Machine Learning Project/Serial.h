#pragma once

#include <iostream>
#include <vector>
#include <Eigen/Dense>

#include "Learnable.h"

using namespace std;
class Serial :
    public Learnable
{
public:
    Serial(vector<Layer*> Layers) :Learnable("Serial"), Layers(Layers) {
        for (auto l : Layers) {
            if (l->isLearnable()) {
                auto p = ((Learnable*)l)->getParam();
                auto g = ((Learnable*)l)->getGrad();

                this->addParam(p, g);
            }
        }
    };

    virtual Eigen::MatrixXd forward(Eigen::MatrixXd panIn) override;
    virtual Eigen::MatrixXd backward(Eigen::MatrixXd preDiff) override;

    virtual ostream& printConfig(ostream& os) override;
    virtual ostream& printConfig(ostream& os, unsigned int level) override;
private:
    vector<Layer*> Layers;
};

