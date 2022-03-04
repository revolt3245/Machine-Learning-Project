#include "CELoss.h"

double CELoss::forward(Eigen::MatrixXd pred, Eigen::MatrixXd actual)
{
	if (type == CELossType::sigmoid) {
		auto batchSize = pred.rows();
		auto outputSize = pred.cols();

		Eigen::MatrixXd ones = Eigen::MatrixXd::Ones(batchSize, outputSize);

		Eigen::MatrixXd positive = (actual.array() * pred.array().log()).matrix();
		Eigen::MatrixXd negative = ((ones - actual).array()
			* (ones - pred).array().log()).matrix();

		this->Diff = -(actual.array() / pred.array() - (ones - actual).array() / (ones - pred).array()).matrix()/batchSize;

		auto cost = -Eigen::MatrixXd::Ones(1, batchSize) * (positive + negative) * Eigen::MatrixXd::Ones(outputSize, 1) / batchSize;
		
		return cost(0);
	}
	else if (type == CELossType::softmax) {
		auto batchSize = pred.rows();
		auto outputSize = pred.cols();

		Eigen::MatrixXd ones = Eigen::MatrixXd::Ones(batchSize, outputSize);

		Eigen::MatrixXd positive = (actual.array() * pred.array().log()).matrix();

		this->Diff = (actual.array() / pred.array()).matrix();

		auto cost = Eigen::MatrixXd::Ones(1, batchSize) * positive * Eigen::MatrixXd::Ones(outputSize, 1) / batchSize;

		return cost(0);
	}
}

Eigen::MatrixXd CELoss::backward()
{
	return this->Diff;
}

ostream& CELoss::printConfig(ostream& os)
{
	if (type == CELossType::sigmoid) {
		os << this->getName() << "(sigmoid)";

		return os;
	}
	else if (type == CELossType::softmax) {
		os << this->getName() << "(softmax)";

		return os;
	}
}
