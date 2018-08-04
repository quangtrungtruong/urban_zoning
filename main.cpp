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

	if (argc < 2) {
		std::cout << "Usage: " << argv[0] << " <dataset or path to dataset> <Optional: city> " << std::endl;
		return 0;
	}

	bool ho_enabled = true;
	bool pairewise_enabled = true;
	double beta = 0.8;
	double anpha= 0.2;
	float gaussian_w = 1.5*pow(10,6);
	float bilateral_w = pow(10,9);
	int iteration = 10;
	int w = 46;
	float bos_acc1, bos_acc2, bos_acc3, bos_acc4, bos_acc5, bos_acc6, sfo_acc1, sfo_acc2, sfo_acc3, sfo_acc4, sfo_acc5, sfo_acc6, nyc_acc1, nyc_acc2, nyc_acc3, nyc_acc4, nyc_acc5, nyc_acc6;

	if (argc>2)
	{
		UrbanObject mo(argv[1], argv[2]);
		mo.RunDenseCRF(ho_enabled, pairewise_enabled, anpha, beta, w, iteration, gaussian_w, bilateral_w, 1);
	} else
	{
		UrbanObject bos(argv[1], "BOS");
		UrbanObject nyc(argv[1], "NYC");
		UrbanObject sfo(argv[1], "SFO");

        for (int i=0; i<10;i++)
        {
            w = 10;
            float param_w = i*pow(10,-6);
            cout << "BOS" << endl;
            bos.RunDenseCRF(true, pairewise_enabled, 0.2, 0.8, w, iteration, gaussian_w, bilateral_w, param_w);
            bos_acc1 = bos.acc;
            bos.RunDenseCRF(false, pairewise_enabled, 0.2, 0.8, w, iteration, gaussian_w, bilateral_w, param_w);
            bos_acc2 = bos.acc;
            cout << endl << "NYC" << endl;
            nyc.RunDenseCRF(true, pairewise_enabled, 0.2, 0.8, w, iteration, gaussian_w, bilateral_w, param_w);
            nyc_acc1 = nyc.acc;
            nyc.RunDenseCRF(false, pairewise_enabled, 0.2, 0.8, w, iteration, gaussian_w, bilateral_w, param_w);
            nyc_acc2 = nyc.acc;
            cout << endl << "SFO" << endl;
            sfo.RunDenseCRF(true, pairewise_enabled, 0.2, 0.8, w, iteration, gaussian_w, bilateral_w, param_w);
            sfo_acc1 = sfo.acc;
            sfo.RunDenseCRF(false, pairewise_enabled, 0.2, 0.8, w, iteration, gaussian_w, bilateral_w, param_w);
            sfo_acc2 = sfo.acc;
			cout << endl << param_w << " " << (bos_acc1 + nyc_acc1 + sfo_acc1) / 3 << " " << (bos_acc2 + nyc_acc2 + sfo_acc2) / 3 << endl;
        }
	}

	return 0;
}

