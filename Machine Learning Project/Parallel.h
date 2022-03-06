#pragma once

#include <iostream>
#include <Eigen/Dense>
#include <vector>

using namespace std;

#include "Learnable.h"
class Parallel :
    public Learnable
{
public:
    Parallel(vector<Layer*> Layers) 
        :Learnable("Parallel"), Layers(Layers), ForkMapping(), JoinMapping() {};
    Parallel(vector<Layer*> Layers, vector<Eigen::MatrixXd> ForkMapping, vector<Eigen::MatrixXd> JoinMapping)
        :Learnable("Parallel"), Layers(Layers), ForkMapping(ForkMapping), JoinMapping(JoinMapping) {};

    virtual Eigen::MatrixXd forward(Eigen::MatrixXd panIn) override;
    virtual Eigen::MatrixXd backward(Eigen::MatrixXd preDiff) override;
    virtual ostream& printConfig(ostream& os) override;
    virtual ostream& printConfig(ostream& os, unsigned int level) override;

private:
    vector<Layer*> Layers;
    vector<Eigen::MatrixXd> ForkMapping;
    vector<Eigen::MatrixXd> JoinMapping;
};