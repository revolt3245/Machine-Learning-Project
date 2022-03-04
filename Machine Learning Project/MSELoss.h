#pragma once

#include <iostream>
#include <Eigen/Dense>

#include "Criterion.h"

using namespace std;

class MSELoss : public Criterion
{
public:
	MSELoss():Criterion("MSELoss") {};

	~MSELoss() {};

	virtual double forward(Eigen::MatrixXd Pred, Eigen::MatrixXd Actual) override;
	virtual Eigen::MatrixXd backward() override;

	virtual ostream& printConfig(ostream& os) override;
private:
	Eigen::MatrixXd Diff;
};

