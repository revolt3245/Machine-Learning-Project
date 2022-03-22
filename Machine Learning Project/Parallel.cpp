#include "Parallel.h"

Eigen::MatrixXd Parallel::forward(Eigen::MatrixXd panIn)
{
	//Fork Mapping
	vector<Eigen::MatrixXd> Fork(0);

	Eigen::MatrixXd res = Eigen::MatrixXd::Zero(panIn.rows(), JoinMapping[0].cols());

	for (auto& f : this->ForkMapping) {
		Fork.push_back(panIn * f);
	}

	for (auto i = 0; i < this->Layers.size(); i++) {
		Fork[i] = this->Layers[i]->forward(Fork[i]);
		res += Fork[i] * JoinMapping[i];
	}

	return res;
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
