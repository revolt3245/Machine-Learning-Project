#pragma once

#include <iostream>
#include <Eigen/Dense>
#include <random>

#include "Layer.h"
#include "Learnable.h"

using namespace std;

class Linear : public Learnable
{
public:
	Linear(size_t input, size_t output) :input(input), output(output), Learnable("Linear", 2) { 
		random_device rd;
		mt19937 rng(rd());

		normal_distribution<> dist(0, 1);

		auto Param = getParam();

		Param[0]->value = Eigen::MatrixXd::NullaryExpr(input, output, [&]() {return dist(rng); });
		Param[1]->value = Eigen::MatrixXd::NullaryExpr(1, output, [&]() {return dist(rng); });

		Param[0]->regL2 = 1.0;
		Param[1]->regL2 = 0.0;
	};

	~Linear() {};

	virtual Eigen::MatrixXd forward(Eigen::MatrixXd panIn) override;
	virtual Eigen::MatrixXd backward(Eigen::MatrixXd preDiff) override;

	Eigen::MatrixXd getWeight();
	Eigen::MatrixXd getBias();

	virtual ostream& printConfig(ostream& os) override;
	virtual ostream& printConfig(ostream& os, unsigned int level) override;
private:
	size_t input;
	size_t output;
};

