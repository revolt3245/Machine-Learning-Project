#include "Learnable.h"

vector<Eigen::MatrixXd*> Learnable::getParam()
{
	return this->param;
}

vector<Eigen::MatrixXd*> Learnable::getGrad()
{
	return this->grad;
}

vector<double> Learnable::getL2Regularization()
{
	return this->regL2;
}

void Learnable::addParam(Eigen::MatrixXd* param)
{
	this->param.push_back(param);
	this->grad.push_back(new Eigen::MatrixXd);
	this->regL2.push_back(0.0);
}

void Learnable::addParam(Eigen::MatrixXd* param, Eigen::MatrixXd* grad)
{
	this->param.push_back(param);
	this->grad.push_back(grad);
	this->regL2.push_back(0.0);
}

void Learnable::addParam(vector<Eigen::MatrixXd*> params)
{
	vector<Eigen::MatrixXd*> gradAdd(params.size(), new Eigen::MatrixXd);
	vector<double> L2Add(params.size());
	this->param.insert(this->param.end(), params.begin(), params.end());

	this->grad.insert(this->grad.end(), gradAdd.begin(), gradAdd.end());
	this->regL2.insert(this->regL2.end(), L2Add.begin(), L2Add.end());
}

void Learnable::addParam(vector<Eigen::MatrixXd*> params, vector<Eigen::MatrixXd*> grads)
{
	vector<double> L2Add(params.size());
	this->param.insert(this->param.end(), params.begin(), params.end());

	this->grad.insert(this->grad.end(), grads.begin(), grads.end());
	this->regL2.insert(this->regL2.end(), L2Add.begin(), L2Add.end());
}

void Learnable::setL2Regularization(vector<double> regL2)
{
	this->regL2 = regL2;
}

void Learnable::setL2Regularization(size_t idx, double regL2)
{
	this->regL2[idx] = regL2;
}
