#include "Criterion.h"

ostream& operator<<(ostream& os, Criterion& obj)
{
	return obj.printConfig(os);
}

string Criterion::getName()
{
	return this->name;
}
