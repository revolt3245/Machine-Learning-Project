#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <random>

#include "hCriterion.h"
#include "hLayer.h"

using namespace std;

int main() {
	Linear L1(3, 4);
	Softmax Soft1;
	CELoss criterion;

	Eigen::MatrixXd X(8, 3);
	Eigen::MatrixXd Y(8, 4);
	X << 0, 0, 0,
		0, 0, 1,
		0, 1, 0,
		0, 1, 1,
		1, 0, 0,
		1, 0, 1,
		1, 1, 0,
		1, 1, 1;

	Y << 1, 0, 0, 0,
		0, 1, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 1, 0,
		0, 0, 0, 1;

	for (int i = 0; i < 10000; i++) {
		auto Pred = L1.forward(X);
		Pred = Soft1.forward(Pred);

		auto cost = criterion.forward(Pred, Y);

		if ((i + 1) % 10 == 0) cout << "epoch " << i + 1 << " : " << cost << "\n";

		auto Diff = criterion.backward();
		Diff = Soft1.backward(Diff);
		L1.backward(Diff);

		L1.step(1e-1);
	}

	cout << Soft1.forward(L1.forward(X)) << "\n";
	return 0;
}
