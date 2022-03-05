#pragma once

#include <iostream>
#include <Eigen/Dense>

using namespace std;

class Layer
{
public:
	Layer() :name(), learnable(false) {};
	Layer(string name) :name(name), learnable(false) {};
	Layer(string name, bool learnable) :name(name), learnable(learnable) {};

	friend class Optimizer;

	virtual Eigen::MatrixXd forward(Eigen::MatrixXd panIn) = 0;
	virtual Eigen::MatrixXd backward(Eigen::MatrixXd preDiff) = 0;

	friend ostream& operator<<(ostream& os, Layer& obj);

	virtual ostream& printConfig(ostream& os) = 0;
	virtual ostream& printConfig(ostream& os, unsigned int level) = 0;

	bool isLearnable();
protected:
	string getName();
private:
	string name;
	bool learnable;
};