#include "Layer.h"

ostream& operator<<(ostream& os, Layer& obj)
{
	return obj.printConfig(os);
}

string Layer::getName()
{
	return this->name;
}
