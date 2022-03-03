#include "MSELoss.h"

ostream& operator<<(ostream& os, MSELoss& obj)
{
    os << "MSELoss";
    return os;
}

double MSELoss::forward(Eigen::MatrixXd Pred, Eigen::MatrixXd Actual)
{
    auto error = Pred - Actual;
    auto outputSize = Pred.cols();
    auto batchSize = Pred.rows();

    this->Diff = error/outputSize/batchSize;

    Eigen::MatrixXd costVector = (error.array() * error.array()).matrix() * Eigen::MatrixXd::Ones(outputSize, 1);
    double cost = (Eigen::MatrixXd::Ones(1, batchSize) * costVector)(0)/outputSize/2/batchSize;

    return cost;
}

Eigen::MatrixXd MSELoss::backward()
{
    return this->Diff;
}
