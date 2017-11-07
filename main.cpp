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
		/*for (int i = 2; i < 10; i++)
			for (int j = 9; j < 100; j++)
				{
					gaussian_w = 1.5*pow(10, 6);
					bilateral_w = pow(10,9);
					anpha = 0.2;
					beta = 0.8;
					cout << endl << i << " " << j <<  " " << gaussian_w << " " << bilateral_w << endl;
					cout << "BOS" << endl;
					bos.RunDenseCRF(true, pairewise_enabled, 0.2, 0.8, w, iteration, gaussian_w, bilateral_w);
					bos_acc1 = bos.acc;
					bos.RunDenseCRF(false, pairewise_enabled, 0.2, 0.8, w, iteration, gaussian_w, bilateral_w);
					bos_acc2 = bos.acc;
					bos.RunDenseCRF(true, pairewise_enabled, 0, 1, w, iteration, gaussian_w, bilateral_w);
					bos_acc3 = bos.acc;
					bos.RunDenseCRF(true, pairewise_enabled, 1, 0, w, iteration, gaussian_w, bilateral_w);
					bos_acc4 = bos.acc;
					bos.RunDenseCRF(false, pairewise_enabled, 1, 0, w, iteration, gaussian_w, bilateral_w);
					bos_acc5 = bos.acc;
					bos.RunDenseCRF(false, pairewise_enabled, 0, 1, w, iteration, gaussian_w, bilateral_w);
					bos_acc6 = bos.acc;
					cout << endl << "NYC" << endl;
					nyc.RunDenseCRF(true, pairewise_enabled, 0.2, 0.8, w, iteration, gaussian_w, bilateral_w);
					nyc_acc1 = nyc.acc;
					nyc.RunDenseCRF(false, pairewise_enabled, 0.2, 0.8, w, iteration, gaussian_w, bilateral_w);
					nyc_acc2 = nyc.acc;
					nyc.RunDenseCRF(true, pairewise_enabled, 0, 1, w, iteration, gaussian_w, bilateral_w);
					nyc_acc3 = nyc.acc;
					nyc.RunDenseCRF(true, pairewise_enabled, 1, 0, w, iteration, gaussian_w, bilateral_w);
					nyc_acc4 = nyc.acc;
					nyc.RunDenseCRF(false, pairewise_enabled, 1, 0, w, iteration, gaussian_w, bilateral_w);
					nyc_acc5 = nyc.acc;
					nyc.RunDenseCRF(false, pairewise_enabled, 0, 1, w, iteration, gaussian_w, bilateral_w);
					nyc_acc6 = nyc.acc;
					cout << endl << "SFO" << endl;
					sfo.RunDenseCRF(true, pairewise_enabled, 0.2, 0.8, w, iteration, gaussian_w, bilateral_w);
					sfo_acc1 = sfo.acc;
					sfo.RunDenseCRF(false, pairewise_enabled, 0.2, 0.8, w, iteration, gaussian_w, bilateral_w);
					sfo_acc2 = sfo.acc;
					sfo.RunDenseCRF(true, pairewise_enabled, 0, 1, w, iteration, gaussian_w, bilateral_w);
					sfo_acc3 = sfo.acc;
					sfo.RunDenseCRF(true, pairewise_enabled, 1, 0, w, iteration, gaussian_w, bilateral_w); 
					sfo_acc4 = sfo.acc;
					sfo.RunDenseCRF(false, pairewise_enabled, 1, 0, w, iteration, gaussian_w, bilateral_w);
					sfo_acc5 = sfo.acc;
					sfo.RunDenseCRF(false, pairewise_enabled, 0, 1, w, iteration, gaussian_w, bilateral_w); 
					sfo_acc6 = sfo.acc;

					cout << endl << (bos_acc1 + nyc_acc1 + sfo_acc1) / 3 << " " << (bos_acc2 + nyc_acc2 + sfo_acc2) / 3 
					<< " " << (bos_acc3+ nyc_acc3 + sfo_acc3) / 3 << " " << (bos_acc4+ nyc_acc4 + sfo_acc4) / 3 << " " 
					<< (bos_acc5+ nyc_acc5 + sfo_acc5) / 3 << " " << (bos_acc6+ nyc_acc6 + sfo_acc6) / 3 << endl;
				}*/

        for (int i=1; i<4;i++)
        {
            w = 10*i;
            float param_w = 0.4;
            cout << "BOS" << endl;
            bos.RunDenseCRF(true, pairewise_enabled, 0.2, 0.8, w, iteration, gaussian_w, bilateral_w, param_w);
            bos_acc1 = bos.acc;
            bos.RunDenseCRF(false, pairewise_enabled, 0.2, 0.8, w, iteration, gaussian_w, bilateral_w, param_w);
            bos_acc2 = bos.acc;
            //bos.RunDenseCRF(true, pairewise_enabled, 0, 1, w, iteration, gaussian_w, bilateral_w);
            //bos_acc3 = bos.acc;
            //bos.RunDenseCRF(true, pairewise_enabled, 1, 0, w, iteration, gaussian_w, bilateral_w);
            //bos_acc4 = bos.acc;
            //bos.RunDenseCRF(false, pairewise_enabled, 1, 0, w, iteration, gaussian_w, bilateral_w);
            //bos_acc5 = bos.acc;
            //bos.RunDenseCRF(false, pairewise_enabled, 0, 1, w, iteration, gaussian_w, bilateral_w);
            //bos_acc6 = bos.acc;
            cout << endl << "NYC" << endl;
            nyc.RunDenseCRF(true, pairewise_enabled, 0.2, 0.8, w, iteration, gaussian_w, bilateral_w, param_w);
            nyc_acc1 = nyc.acc;
            nyc.RunDenseCRF(false, pairewise_enabled, 0.2, 0.8, w, iteration, gaussian_w, bilateral_w, param_w);
            nyc_acc2 = nyc.acc;
            //nyc.RunDenseCRF(true, pairewise_enabled, 0, 1, w, iteration, gaussian_w, bilateral_w);
            //nyc_acc3 = nyc.acc;
            //nyc.RunDenseCRF(true, pairewise_enabled, 1, 0, w, iteration, gaussian_w, bilateral_w);
            //nyc_acc4 = nyc.acc;
            //nyc.RunDenseCRF(false, pairewise_enabled, 1, 0, w, iteration, gaussian_w, bilateral_w);
            //nyc_acc5 = nyc.acc;
            //nyc.RunDenseCRF(false, pairewise_enabled, 0, 1, w, iteration, gaussian_w, bilateral_w);
            //nyc_acc6 = nyc.acc;
            cout << endl << "SFO" << endl;
            sfo.RunDenseCRF(true, pairewise_enabled, 0.2, 0.8, w, iteration, gaussian_w, bilateral_w, param_w);
            sfo_acc1 = sfo.acc;
            sfo.RunDenseCRF(false, pairewise_enabled, 0.2, 0.8, w, iteration, gaussian_w, bilateral_w, param_w);
            sfo_acc2 = sfo.acc;
            //sfo.RunDenseCRF(true, pairewise_enabled, 0, 1, w, iteration, gaussian_w, bilateral_w);
            //sfo_acc3 = sfo.acc;
            //sfo.RunDenseCRF(true, pairewise_enabled, 1, 0, w, iteration, gaussian_w, bilateral_w);
            //sfo_acc4 = sfo.acc;
            //sfo.RunDenseCRF(false, pairewise_enabled, 1, 0, w, iteration, gaussian_w, bilateral_w);
            //sfo_acc5 = sfo.acc;
            //sfo.RunDenseCRF(false, pairewise_enabled, 0, 1, w, iteration, gaussian_w, bilateral_w);
            //sfo_acc6 = sfo.acc;

            //cout << endl << (bos_acc1 + nyc_acc1 + sfo_acc1) / 3 << " " << (bos_acc2 + nyc_acc2 + sfo_acc2) / 3
            //<< " " << (bos_acc3+ nyc_acc3 + sfo_acc3) / 3 << " " << (bos_acc4+ nyc_acc4 + sfo_acc4) / 3 << " "
            //<< (bos_acc5+ nyc_acc5 + sfo_acc5) / 3 << " " << (bos_acc6+ nyc_acc6 + sfo_acc6) / 3 << endl;
            //cout << endl << (bos_acc1 + nyc_acc1 + sfo_acc1) / 3 << " "  << (bos_acc3+ nyc_acc3 + sfo_acc3) / 3 << " " << (bos_acc4+ nyc_acc4 + sfo_acc4) / 3 << endl;
            cout << endl << w << " " << (bos_acc1 + nyc_acc1 + sfo_acc1) / 3 << " " << (bos_acc2 + nyc_acc2 + sfo_acc2) / 3 << endl;
        }
	}

	return 0;
}

