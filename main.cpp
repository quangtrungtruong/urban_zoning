#include "urban_object.h"
#include "iostream"
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
	double beta;
	double anpha;
	float gaussian_w = 1.5*pow(10,6);
	float bilateral_w = pow(10,9);
	int iteration = 10;
	int size_window_term4;
    float weight_term4;
	float bos_acc1, bos_acc2, bos_acc3, bos_acc4, bos_acc5, bos_acc6, bos_acc7, bos_acc8, sfo_acc1, sfo_acc2, sfo_acc3, sfo_acc4, sfo_acc5, sfo_acc6, sfo_acc7, sfo_acc8, nyc_acc1, nyc_acc2, nyc_acc3, nyc_acc4, nyc_acc5, nyc_acc6, nyc_acc7, nyc_acc8;

	if (argc>2)
	{
		float mo_acc1, mo_acc2, mo_acc3, mo_acc4, mo_acc5, mo_acc6, mo_acc7, mo_acc8;
		UrbanObject mo(argv[1], argv[2]);
        size_window_term4 = 46;
        weight_term4 = 2;
        float param_w = 6*pow(10,-9);
        anpha = 0.73;
        beta = 1.0 - anpha;
        mo.RunDenseCRF(true, pairewise_enabled, anpha, beta, size_window_term4, iteration, gaussian_w, bilateral_w, param_w, weight_term4);
 	} else
	{
		UrbanObject bos(argv[1], "BOS");
		UrbanObject nyc(argv[1], "NYC");
		UrbanObject sfo(argv[1], "SFO");
        cout << "Test existing term2, origin images";
        for (int i=1; i<2;i++)
        {
            size_window_term4 = 46;
            weight_term4 = 2;
            float param_w = 6*pow(10,-9);
            cout << "BOS" << endl;
            anpha = 0.73;
            beta = 1.0 - anpha;
            bos.RunDenseCRF(true, pairewise_enabled, anpha, beta, size_window_term4, iteration, gaussian_w, bilateral_w, param_w, weight_term4);
            bos_acc1 = bos.acc;
            bos.RunDenseCRF(false, pairewise_enabled, anpha, beta, size_window_term4, iteration, gaussian_w, bilateral_w, param_w, weight_term4);
            bos_acc2 = bos.acc;
            cout << endl << "NYC" << endl;
            nyc.RunDenseCRF(true, pairewise_enabled, anpha, beta, size_window_term4, iteration, gaussian_w, bilateral_w, param_w, weight_term4);
            nyc_acc1 = nyc.acc;
            nyc.RunDenseCRF(false, pairewise_enabled, anpha, beta, size_window_term4, iteration, gaussian_w, bilateral_w, param_w, weight_term4);
            nyc_acc2 = nyc.acc;
            cout << endl << "SFO" << endl;
            sfo.RunDenseCRF(true, pairewise_enabled, anpha, beta, size_window_term4, iteration, gaussian_w, bilateral_w, param_w, weight_term4);
            sfo_acc1 = sfo.acc;
            sfo.RunDenseCRF(false, pairewise_enabled, anpha, beta, size_window_term4, iteration, gaussian_w, bilateral_w, param_w, weight_term4);
            sfo_acc2 = sfo.acc;

            cout << endl << "----- Accuracy: "  << weight_term4  << " "<< (bos_acc1 + nyc_acc1 + sfo_acc1) / 3 << " "<< (bos_acc2 + nyc_acc2 + sfo_acc2) / 3  << endl;
        }
	}

	return 0;
}

