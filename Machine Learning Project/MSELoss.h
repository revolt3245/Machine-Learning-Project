#pragma once

#include <iostream>
#include <Eigen/Dense>

using namespace std;

class MSELoss
{
public:
	MSELoss() {};

	~MSELoss() {};

	friend ostream& operator<<(ostream& os, MSELoss& obj);

	double forward(Eigen::MatrixXd Pred, Eigen::MatrixXd Actual);
	Eigen::MatrixXd backward();
private:
	Eigen::MatrixXd Diff;
};

