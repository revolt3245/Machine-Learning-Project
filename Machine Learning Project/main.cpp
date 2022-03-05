#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <random>
#include <crtdbg.h>

#if _DEBUG
#define new new(_NORMAL_BLOCK, __FILE__, __LINE__)
#endif

#include "hCriterion.h"
#include "hLayer.h"
#include "hOptimizer.h"

using namespace std;

int main() {
	Linear L1(2, 4), L2(4, 1);
	ReLU R1;
	Sigmoid Sig1;
	Serial Net({ &L1, &R1, &L2, &Sig1 });
	CELoss criterion(CELoss::CELossType::sigmoid);
	SGD optim(1e-2, 1e-4);

	Eigen::MatrixXd X(4, 2);
	Eigen::MatrixXd Y(4, 1);
	X << 0, 0,
		0, 1,
		1, 0,
		1, 1;

	Y << 0, 1, 1, 0;

	for (int i = 0; i < 10000; i++) {
		auto Pred = Net.forward(X);

		auto cost = criterion.forward(Pred, Y);

		if ((i + 1) % 10 == 0) cout << "epoch " << i + 1 << " : " << cost << "\n";

		auto Diff = criterion.backward();
		Net.backward(Diff);

		optim.step(&Net);
	}

	cout << Net.forward(X) << "\n";

	return 0;
}
