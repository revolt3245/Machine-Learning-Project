#include "Serial.h"

Eigen::MatrixXd Serial::forward(Eigen::MatrixXd panIn)
{
    Eigen::MatrixXd Output = panIn;
    for (auto l : this->Layers) {
        Output = l->forward(Output);
    }
    return Output;
}

Eigen::MatrixXd Serial::backward(Eigen::MatrixXd preDiff)
{
    Eigen::MatrixXd Diff = preDiff;
    auto cpyLayers = Layers;
    reverse(cpyLayers.begin(), cpyLayers.end());

    for (auto l : cpyLayers) {
        Diff = l->backward(Diff);
    }
    return Diff;
}

ostream& Serial::printConfig(ostream& os)
{
    os << this->getName() << " (" << "\n";
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
        os << "    ";
    }
    os << this->getName() << " (" << "\n";
    for (auto l : this->Layers) {
        l->printConfig(os, level + 1);
        os << "\n";
    }
    for (auto i = 0; i < level; i++) {
        os << "    ";
    }
    os << ")";

    return os;
}
