#pragma once

#include <iostream>
#include <Eigen/Dense>
#include <random>

#include "Layer.h"

using namespace std;

class Linear : public Layer
{
public:
	Linear(size_t input, size_t output) :input(input), output(output), Layer("Linear") { 
		random_device rd;
		mt19937 rng(rd());

		normal_distribution<> dist(0, 1);

		//weight = Eigen::MatrixXd::Random(input, output);  
		//bias = Eigen::MatrixXd::Random(1, output); 
		weight = Eigen::MatrixXd::NullaryExpr(input, output, [&]() {return dist(rng); });
		bias = Eigen::MatrixXd::NullaryExpr(1, output, [&]() {return dist(rng); });
	};

	~Linear() {};

	virtual Eigen::MatrixXd forward(Eigen::MatrixXd panIn) override;
	virtual Eigen::MatrixXd backward(Eigen::MatrixXd preDiff) override;

	void step(double learningRate);

	Eigen::MatrixXd getWeight();
	Eigen::MatrixXd getBias();
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

