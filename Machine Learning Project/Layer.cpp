#include "Layer.h"

ostream& operator<<(ostream& os, Layer& obj)
{
	return obj.printConfig(os, 0);
}

bool Layer::isLearnable()
{
	return this->learnable;
}

string Layer::getName()
{
	return this->name;
}
