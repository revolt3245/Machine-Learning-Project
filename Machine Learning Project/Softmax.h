#pragma once

#include <iostream>
#include <Eigen/Dense>

using namespace std;

#include "Layer.h"
class Softmax :
    public Layer
{
public:
    Softmax() :Layer("Softmax") {};

    ~Softmax() {};

    // Layer��(��) ���� ��ӵ�
    virtual Eigen::MatrixXd forward(Eigen::MatrixXd panIn) override;
    virtual Eigen::MatrixXd backward(Eigen::MatrixXd preDiff) override;

    virtual ostream& printConfig(ostream& os) override;
    virtual ostream& printConfig(ostream& os, unsigned int level) override;
private:
    Eigen::MatrixXd panOut;
};

