#pragma once
class Test
{
public:
	Test() {};
	~Test() {};

	virtual double func(double x) = 0;
	virtual int func(int x) = 0;
private:

};

class Test1 :public Test {
public:
	Test1() :Test() {};

	~Test1() {};
	virtual double func(double x) {
		return x;
	}

	virtual int func(int x) { return; }
private:
};

