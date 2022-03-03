#pragma once

#include <iostream>
#include <Eigen/Dense>

using namespace std;

class Layer
{
public:
	Layer() :name() {};
	Layer(string name) :name(name) {};

	friend class Optimizer;

	virtual Eigen::MatrixXd forward(Eigen::MatrixXd panIn) = 0;
	virtual Eigen::MatrixXd backward(Eigen::MatrixXd preDiff) = 0;

	friend ostream& operator<<(ostream& os, Layer& obj);
protected:
	virtual ostream& printConfig(ostream& os) = 0;

	string getName();
private:
	string name;
};