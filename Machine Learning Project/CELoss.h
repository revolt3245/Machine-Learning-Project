#pragma once

#include <iostream>
#include <Eigen/Dense>

#include "Criterion.h"

using namespace std;

class CELoss : public Criterion
{
public:
	enum class CELossType {
		sigmoid,
		softmax
	};

	CELoss(CELossType type = CELossType::softmax) :Criterion("CELoss"), type(type) {};

	~CELoss() {};
	
	virtual double forward(Eigen::MatrixXd pred, Eigen::MatrixXd actual) override;
	virtual Eigen::MatrixXd backward() override;
protected:
	virtual ostream& printConfig(ostream& os) override;
private:
	CELossType type;
	Eigen::MatrixXd Diff;
};

