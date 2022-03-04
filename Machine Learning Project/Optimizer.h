#pragma once

#include <iostream>
#include <Eigen/Dense>

#include "Learnable.h"

using namespace std;

class Optimizer
{
public:
	Optimizer(double learningRate = 1e-2, double regL2 = 1e-4) :
		learningRate(learningRate), regL2(regL2) {};

	~Optimizer() {};
	virtual void step(Learnable* obj) = 0;

	double getLearningRate();
	double getL2Regularization();

	void setLearningRate(double learningRate);
	void setL2Regularization(double regL2);
protected:
	vector<Eigen::MatrixXd*> getLearnableParam(Learnable* obj);
	vector<Eigen::MatrixXd*> getLearnableGrad(Learnable* obj);
	vector<double> getLearnableL2Regularization(Learnable* obj);
private:
	double learningRate;
	double regL2;
};

