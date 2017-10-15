#include "urban_object.h"
#include "iostream"

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

	bool ho_enabled = false;
	bool cooc_enabled = false;
	if (argc == 4) {
		if (strcmp(argv[3], "ho") == 0)
			ho_enabled = true;
		if (strcmp(argv[3], "cooc") == 0)
			cooc_enabled = true;
	}
	else if (argc == 5) {
		ho_enabled = true;
		cooc_enabled = true;
	}
	mo.RunDenseCRF(ho_enabled, cooc_enabled);
	return 0;

	/*if (strcmp(argv[2], "unary_sscnet") == 0) {
		if (argc < 5) {
			std::cout << "unary_sscnet <pose file>  <probability file>" << std::endl;
			return 0;
		}

		std::string poseFile = argv[3];
		std::string probFile = argv[4];
		mo.MakeUnaryFromSSCNet(poseFile, probFile);
		return 0;
	}*/
}
