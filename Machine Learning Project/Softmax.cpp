#include "Softmax.h"

Eigen::MatrixXd Softmax::forward(Eigen::MatrixXd panIn)
{
	auto inputSize = panIn.cols();
	Eigen::MatrixXd expPanIn = panIn.array().exp().matrix();

	Eigen::MatrixXd Sum = expPanIn * Eigen::MatrixXd::Ones(inputSize, 1) * Eigen::MatrixXd::Ones(1, inputSize);

	this->panOut = (expPanIn.array() / Sum.array()).matrix();

	return panOut;
}

Eigen::MatrixXd Softmax::backward(Eigen::MatrixXd preDiff)
{
	auto Diff = panOut.array() * (1 - panOut.array());

	return (Diff * preDiff.array()).matrix();
}

ostream& Softmax::printConfig(ostream& os)
{
	os << this->getName();

	return os;
}
