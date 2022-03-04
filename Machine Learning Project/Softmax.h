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
protected:
    virtual ostream& printConfig(ostream& os) override;
private:
    Eigen::MatrixXd Diff;
};

