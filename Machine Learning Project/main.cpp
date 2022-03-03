#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <random>

#include "Linear.h"
#include "MSELoss.h"

using namespace std;

int main() {
	Linear L(2, 1);
	MSELoss criteria;
	random_device rd;
	mt19937 rng(rd());

	normal_distribution<> dist(0, 1);

	Eigen::MatrixXd X = Eigen::MatrixXd::NullaryExpr(10000, 2, [&]() {return dist(rng); });
	auto A = Eigen::MatrixXd(2, 1);
	auto B = 2.0;

	A << 1.0, 2.0;

	Eigen::MatrixXd Y = X * A + B * Eigen::MatrixXd::Ones(10000, 1) + Eigen::MatrixXd::NullaryExpr(10000, 1, [&]() {return dist(rng); }) * 0.001;

	vector<Eigen::MatrixXd> XBatch(0), YBatch(0);

	for (int i = 0; i < 100; i++) {
		XBatch.push_back(X(Eigen::seq(i * 100, i * 100 + 99), Eigen::all));
		YBatch.push_back(Y(Eigen::seq(i * 100, i * 100 + 99), Eigen::all));
	}

	for (int i = 0; i < 100; i++) {
		double mean_loss = 0;
		for (int j = 0; j < 100; j++) {
			auto Pred = L.forward(XBatch[j]);
			auto loss = criteria.forward(Pred, YBatch[j]);

			mean_loss += loss;
			auto Diff = criteria.backward();
			L.backward(Diff);

			L.step(1e-3);
		}

		mean_loss /= 100;
		if ((i + 1) % 10 == 0)cout << "epoch " << i + 1 << " : " << mean_loss << "\n";
	}

	cout << "\n" << "weight" << "\n";
	cout << L.getWeight() << "\n";
	cout << "bias" << "\n";
	cout << L.getBias() << "\n";
	/*
	cout << "\n" << "predict" << "\n";
	cout << L.forward(X) << "\n";
	*/
	return 0;
}
