#include "Serial.h"

Eigen::MatrixXd Serial::forward(Eigen::MatrixXd panIn)
{
    return Eigen::MatrixXd();
}

Eigen::MatrixXd Serial::backward(Eigen::MatrixXd preDiff)
{
    return Eigen::MatrixXd();
}

ostream& Serial::printConfig(ostream& os)
{
    os << "Serial (" << "\n";
    for (auto l : this->Layers) {
        l->printConfig(os, 1);
        os << "\n";
    }
    os << ")";

    return os;
}

ostream& Serial::printConfig(ostream& os, unsigned int level)
{
    for (auto i = 0; i < level; i++) {
        os << "\t";
    }
    os << "Serial (" << "\n";
    for (auto l : this->Layers) {
        l->printConfig(os, level + 1);
        os << "\n";
    }
    for (auto i = 0; i < level; i++) {
        os << "\t";
    }
    os << ")";

    return os;
}
