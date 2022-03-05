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
