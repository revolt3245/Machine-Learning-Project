#pragma once

#include <iostream>
#include <Eigen/Dense>

using namespace std;

class Criterion
{
public:
	Criterion(string name) :name(name) {};

	~Criterion() {};

	friend ostream& operator<<(ostream& os, Criterion& obj);
	virtual double forward(Eigen::MatrixXd Pred, Eigen::MatrixXd Actual) = 0;
	virtual Eigen::MatrixXd backward() = 0;
protected:
	virtual ostream& printConfig(ostream& os) = 0;

	string getName();
private:
	string name;
};

