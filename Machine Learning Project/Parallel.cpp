#include "Parallel.h"

Eigen::MatrixXd Parallel::forward(Eigen::MatrixXd panIn)
{
	return Eigen::MatrixXd();
}

Eigen::MatrixXd Parallel::backward(Eigen::MatrixXd preDiff)
{
	return Eigen::MatrixXd();
}

ostream& Parallel::printConfig(ostream& os)
{
	os << this->getName() << " (\n";
	for (auto l : this->Layers) {
		l->printConfig(os, 1);
		os << "\n";
	}
	os << ")";

	return os;
}

ostream& Parallel::printConfig(ostream& os, unsigned int level)
{
	for (auto i = 0; i < level; i++) {
		os << "    ";
	}
	os << this->getName() << " (\n";
	for (auto l : this->Layers) {
		l->printConfig(os, level + 1);
		os << "\n";
	}
	os << ")";

	return os;
}
