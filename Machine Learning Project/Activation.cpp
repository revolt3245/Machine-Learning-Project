#include "Activation.h"

Eigen::MatrixXd Activation::getPanOut()
{
	return this->panOut;
}

void Activation::setPanOut(Eigen::MatrixXd panOut)
{
	this->panOut = panOut;
}
