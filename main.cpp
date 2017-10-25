#include "urban_object.h"
#include "iostream"
#include <string.h>
#include <opencv2/opencv.hpp>
#include <fstream>

/**
*  Note: this source code uses STL set and map. It is good to keep in mind
*  that for map, the operator[] will automatically add an item to map
*  if the key is not found.
*  This can be confusing when the code needs to be debugged.
*/

int main(int argc, char* argv[]) {

	if (argc < 3) {
		std::cout << "Usage: " << argv[0] << " <dataset or path to dataset> <city> " << std::endl;
		return 0;
	}
	UrbanObject mo(argv[1], argv[2]);
	bool ho_enabled = true;
	bool pairewise_enabled = true;
	double anpha = 0;
	double beta=1;
	int w = 10;
	int iteration = 10;
	mo.RunDenseCRF(ho_enabled, pairewise_enabled, anpha, beta, w, iteration);

	ho_enabled = true; pairewise_enabled = true; anpha = 1; beta = 0; w = 10; iteration = 10;
	mo.RunDenseCRF(ho_enabled, pairewise_enabled, anpha, beta, w, iteration);

	ho_enabled = true; pairewise_enabled = true; anpha = 0.467; beta = 0.633; w = 10; iteration = 10;
	mo.RunDenseCRF(ho_enabled, pairewise_enabled, anpha, beta, w, iteration);

	ho_enabled = true; pairewise_enabled = true; anpha = 0.467; beta = 0.633; w = 10; iteration = 40;
	mo.RunDenseCRF(ho_enabled, pairewise_enabled, anpha, beta, w, iteration);

	ho_enabled = false; pairewise_enabled = false; anpha = 0.467; beta = 0.633; w = 10; iteration = 10;
	mo.RunDenseCRF(ho_enabled, pairewise_enabled, anpha, beta, w, iteration);

	ho_enabled = false; pairewise_enabled = false; anpha = 1; beta = 0; w = 10; iteration = 10;
	mo.RunDenseCRF(ho_enabled, pairewise_enabled, anpha, beta, w, iteration);

	ho_enabled = false; pairewise_enabled = false; anpha = 0; beta = 1; w = 10; iteration = 10;
	mo.RunDenseCRF(ho_enabled, pairewise_enabled, anpha, beta, w, iteration);

	ho_enabled = false; pairewise_enabled = true; anpha = 1; beta = 0; w = 10; iteration = 10;
	mo.RunDenseCRF(ho_enabled, pairewise_enabled, anpha, beta, w, iteration);

	ho_enabled = true; pairewise_enabled = true; anpha = 0; beta = 1; w = 10; iteration = 10;
	mo.RunDenseCRF(ho_enabled, pairewise_enabled, anpha, beta, w, iteration);

	return 0;
}

