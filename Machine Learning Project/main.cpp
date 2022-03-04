#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <random>

#include "Linear.h"
#include "Sigmoid.h"
#include "CELoss.h"

using namespace std;

int main() {
	Linear L1(2, 1);
	Sigmoid Sig1;
	CELoss criterion(CELoss::CELossType::sigmoid);

	Eigen::MatrixXd X(4, 2);
	Eigen::MatrixXd Y(4, 1);
	X << 0, 0,
		0, 1,
		1, 0,
		1, 1;

	Y << 0,
		1,
		1,
		1;

	for (int i = 0; i < 100000; i++) {
		auto Pred = L1.forward(X);
		Pred = Sig1.forward(Pred);

		auto cost = criterion.forward(Pred, Y);

		if ((i + 1) % 10 == 0) cout << "epoch " << i + 1 << " : " << cost << "\n";

		auto Diff = criterion.backward();
		Diff = Sig1.backward(Diff);
		L1.backward(Diff);

		L1.step(1e-2);
	}

	cout << Sig1.forward(L1.forward(X)) << "\n";
	return 0;
}
