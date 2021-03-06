#include "SGD.h"

void SGD::step(Learnable* obj)
{
	/*
	auto Param = this->getLearnableParam(obj);
	auto Grad = this->getLearnableGrad(obj);
	auto learnableL2 = this->getLearnableL2Regularization(obj);

	auto lr = this->getLearningRate();
	auto l2 = this->getL2Regularization();
	for (int i = 0; i < Param.size(); i++) {
		auto localL2 = learnableL2[i] * l2;
		*Param[i] -= lr * (localL2 * (*Param[i]) + *Grad[i]);
	}
	*/

	auto Param = this->getLearnableParam(obj);

	auto lr = this->getLearningRate();
	auto l2 = this->getL2Regularization();

	for (auto& p : Param) {
		auto localL2 = p->regL2 * l2;
		p->value -= lr * (localL2 * p->value + p->grad);
	}
}
