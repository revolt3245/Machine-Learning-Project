#pragma once

#include <iostream>
#include <Eigen/Dense>

using namespace std;

class Parameter
{
public:
	Parameter();
	Parameter(Eigen::MatrixXd value);

	~Parameter();

	Parameter operator+(Parameter& obj);
	Parameter operator*(Parameter& obj);
	Parameter operator-(Parameter& obj);


private:
	Eigen::MatrixXd value;
	Eigen::MatrixXd grad;
};

