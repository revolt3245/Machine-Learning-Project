#pragma once

#include <iostream>
#include <Eigen/Dense>

using namespace std;

class Linear
{
public:
	Linear(size_t input, size_t output) :input(input), output(output) { weight = Eigen::MatrixXd::Random(input, output);  bias = Eigen::MatrixXd::Random(1, output); };

	~Linear() {};

	friend ostream& operator<<(ostream& os, Linear& obj);

	Eigen::MatrixXd forward(Eigen::MatrixXd panIn);
	Eigen::MatrixXd backward(Eigen::MatrixXd preDiff);

	void step(double learningRate);
private:
	Eigen::MatrixXd weight;
	Eigen::MatrixXd bias;

	Eigen::MatrixXd weightGrad;
	Eigen::MatrixXd biasGrad;

	size_t input;
	size_t output;
};

