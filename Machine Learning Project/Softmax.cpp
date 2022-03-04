#include "Softmax.h"

Eigen::MatrixXd Softmax::forward(Eigen::MatrixXd panIn)
{
	return Eigen::MatrixXd();
}

Eigen::MatrixXd Softmax::backward(Eigen::MatrixXd preDiff)
{
	return Eigen::MatrixXd();
}

ostream& Softmax::printConfig(ostream& os)
{
	os << this->getName();

	return os;
}
