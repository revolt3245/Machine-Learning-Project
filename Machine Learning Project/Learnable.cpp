#include "Learnable.h"

vector<Eigen::MatrixXd*> Learnable::getParam()
{
	return this->param;
}

vector<Eigen::MatrixXd*> Learnable::getGrad()
{
	return this->grad;
}

void Learnable::addParam(Eigen::MatrixXd* param)
{
	this->param.push_back(param);
	this->grad.push_back(new Eigen::MatrixXd);
}

void Learnable::addParam(vector<Eigen::MatrixXd*> params)
{
	vector<Eigen::MatrixXd*> gradAdd(params.size(), new Eigen::MatrixXd);
	this->param.insert(this->param.end(), params.begin(), params.end());

	this->grad.insert(this->grad.end(), gradAdd.begin(), gradAdd.end());
}
