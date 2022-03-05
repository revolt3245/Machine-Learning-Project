#include "Learnable.h"

vector<Parameter*> Learnable::getParam()
{
	return this->param;
}

void Learnable::addParam(Parameter* param)
{
	this->param.push_back(param);
}

void Learnable::addParam(vector<Parameter*> params)
{
	this->param.insert(this->param.end(), params.begin(), params.end());
}

void Learnable::allocParam()
{
	for (auto& p : param) {
		p = new Parameter;
	}
}

void Learnable::deleteParam()
{
	for (auto& p : param) {
		delete p;
	}
}
