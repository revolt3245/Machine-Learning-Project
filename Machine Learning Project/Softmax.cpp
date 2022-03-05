#include "Softmax.h"

Eigen::MatrixXd Softmax::forward(Eigen::MatrixXd panIn)
{
	auto inputSize = panIn.cols();
	Eigen::MatrixXd expPanIn = panIn.array().exp().matrix();

	Eigen::MatrixXd Sum = expPanIn * Eigen::MatrixXd::Ones(inputSize, 1) * Eigen::MatrixXd::Ones(1, inputSize);

	this->setPanOut((expPanIn.array() / Sum.array()).matrix());

	return this->getPanOut();
}

Eigen::MatrixXd Softmax::backward(Eigen::MatrixXd preDiff)
{
	auto Diff = this->getPanOut().array() * (1 - this->getPanOut().array());

	return (Diff * preDiff.array()).matrix();
}

ostream& Softmax::printConfig(ostream& os)
{
	os << this->getName();

	return os;
}

ostream& Softmax::printConfig(ostream& os, unsigned int level)
{
	for (auto i = 0; i < level; i++) {
		os << "\t";
	}
	os << this->getName();

	return os;
}
