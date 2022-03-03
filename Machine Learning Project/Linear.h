#pragma once

#include <iostream>
#include <Eigen/Dense>

#include "Layer.h"

using namespace std;

class Linear : public Layer
{
public:
	Linear(size_t input, size_t output) :input(input), output(output), Layer("Linear") { weight = Eigen::MatrixXd::Random(input, output);  bias = Eigen::MatrixXd::Random(1, output); };

	~Linear() {};

	virtual Eigen::MatrixXd forward(Eigen::MatrixXd panIn) override;
	virtual Eigen::MatrixXd backward(Eigen::MatrixXd preDiff) override;
protected:
	virtual ostream& printConfig(ostream& os) override;
private:
	Eigen::MatrixXd weight;
	Eigen::MatrixXd bias;

	Eigen::MatrixXd weightGrad;
	Eigen::MatrixXd biasGrad;

	size_t input;
	size_t output;
};

