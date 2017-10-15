/* ----------------------------------------------------------------------------
 *
 * Copyright (c) 2016, Lucas Kahlert <lucas.kahlert@tu-dresden.de>
 * Copyright (c) 2012, Vibhav Vineet
 * Copyright (c) 2011, Philipp Krähenbühl
 *
 * All rights reserved.
 *
 * Permission is hereby granted, free of charge, to use this software for
 * evaluation and research purposes.
 *
 * This license does not allow this software to be used in a commercial
 * context.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *   * Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *   * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *
 *   * Neither the name of the Stanford University or Technical University of
 *     Dresden nor the names of its contributors may be used to endorse or
 *     promote products derived from this software without specific prior.
 *
 * THIS SOFTWARE IS PROVIDED BY Lucas Kahlert AND CONTRIBUTORS "AS IS" AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL Lucas Kahlert OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
 * DAMAGE.
 * ------------------------------------------------------------------------- */

#include <fstream>
#include <iostream>
#include <cassert>
#include <cmath>

#ifdef _WIN32
	#include <io.h>
	#include <Windows.h>
#endif

#include "image.h"
#include "higherorder.h"


using namespace std;
using namespace ale;


LDataset* dataset = nullptr;

int num_of_labels = 0;
int num_files = 0;

float *dataCost      = nullptr;
float *cooc_unary    = nullptr;
float *cooc_pairwise = nullptr;

int imagewidth;
int imageheight;
int K;

//char *fileNameL = nullptr;

unsigned char *final_labels = nullptr;
unsigned char *im_orig      = nullptr;

// ho_terms
int num_of_layers;
char const **layers_dir  = nullptr;
char const *ho_stats_pot = nullptr;
char const *ho_seg_ext   = nullptr;
char const *ho_sts_ext   = nullptr;

// det_terms
char const *det_seg_dir = nullptr;
char const *det_bb_dir  = nullptr;
char const *det_seg_ext = nullptr;
char const *det_bb_ext  = nullptr;


void del_meminit()
{
    delete []dataCost;

    delete []final_labels;
    delete []cooc_unary;
    delete []cooc_pairwise;

    delete[] im_orig;
}


void mem_init(const char *fileName)
{
    string completeFilename = string(dataset->imageFolder) + fileName + dataset->imageExtension;
    LRgbImage rgbImage(completeFilename.c_str());
    imageheight = rgbImage.GetHeight();
    imagewidth = rgbImage.GetWidth();

    if (imageheight == 0 || imagewidth == 0) {
        cerr << "Warn: " << "Image " << completeFilename << " is empty" << endl;
    } else {
        dataCost 	  = new float[imagewidth*imageheight*num_of_labels];
        final_labels  = new unsigned char[imagewidth*imageheight];
        cooc_unary 	  = new float[num_of_labels];
        cooc_pairwise = new float[num_of_labels*num_of_labels];

        memset(dataCost,      0, imagewidth * imageheight * num_of_labels * sizeof(float));
        memset(cooc_unary,    0, num_of_labels * sizeof(float));
        memset(cooc_pairwise, 0, num_of_labels * num_of_labels * sizeof(float));
        memset(final_labels,  0, imagewidth*imageheight * sizeof(unsigned char));

        loadFiles(fileName);

        // Copy original image
        im_orig = new unsigned char[3 * imagewidth * imageheight];
        unsigned char *im = rgbImage.GetData();

        for (int i = 0; i < imageheight; i++) {
            for (int j = 0; j < imagewidth; j++) {
                im_orig[3 * ((imageheight - (i + 1)) * imagewidth + j) + 0] = im[3 * (i * imagewidth + j) + 0];
                im_orig[3 * ((imageheight - (i + 1)) * imagewidth + j) + 1] = im[3 * (i * imagewidth + j) + 1];
                im_orig[3 * ((imageheight - (i + 1)) * imagewidth + j) + 2] = im[3 * (i * imagewidth + j) + 2];
            }
        }
    }
}


void loadFiles(const char *fileName)
{
    // char *filename = new char[200];
    // sprintf(filename, "eTRIMS/Result/Dense/%s.c_unary", fileName);
    // // read unary potential
    // ProbImage im_unary;
    // im_unary.decompress(filename);

    // float *temp_data = new float[imagewidth*imageheight*num_of_labels];
    // for(int i = 0; i < imagewidth*imageheight*num_of_labels; i++)
    // 	temp_data[i] = (im_unary.data())[i];

    // for(int i = 0; i < imagewidth*imageheight*num_of_labels; i++)
    // {
    // 	if(temp_data[i] == 0.0)	temp_data[i] = 0.00001;
    // 	if(temp_data[i] == 1.0) temp_data[i] = 0.9998;
    // 	dataCost[i] = -3.0 * log(temp_data[i]);
    // }

    // delete []temp_data;

    // -------------------------------------------
    // start ALE boosted per pixel unary potential
    // -------------------------------------------

    // Uncomment following lines if you want to use unary potentials evaluated using ALE

    double *classifier_val_reverse = new double[imagewidth*imageheight*num_of_labels];
    double *classifier_val = new double[imagewidth*imageheight*num_of_labels];

    for(int i = 0; i < imagewidth * imageheight * num_of_labels; i++) {
        classifier_val_reverse[i] = 0.0; classifier_val[i] = 0.0;
    }

    string filename = string(dataset->denseFolder) + fileName + ".dns";
    FILE *fp = fopen(filename.c_str(), "rb");

    if (fp == nullptr) {
        cerr << "Error: Cannot open file '" << filename << "'" << endl;
        return;
    }

    assert(fp != nullptr && "File must exist");

    int temp[3];
    fread(temp, sizeof(int), 3, fp);
    fread(classifier_val_reverse, sizeof(double), imagewidth*imageheight*num_of_labels, fp);
    fclose(fp);


    // start
    for(int k = 0; k < num_of_labels; k++) {
        for(int i = 0; i < imageheight; i++) {
            for(int j = 0; j < imagewidth; j++) {
                classifier_val[num_of_labels*((imageheight-(1+i))*imagewidth+j)+k] = classifier_val_reverse[num_of_labels*(i*imagewidth+j)+k];
            }
        }
    }

    // calculating the per pixel dataCost
    double sum = 0.0;
    for(int i = 0; i < imageheight; i++) {
        for(int j = 0; j < imagewidth; j++) {
            sum = 0.0;
            double data_cost_val = 0.0;
            for(int l = 0; l < num_of_labels; l++) {
                sum += exp(classifier_val[num_of_labels*(i*imagewidth+j)+l]);
            }

            for(int l = 0; l < num_of_labels; l++) {
                data_cost_val = -1.0 * log(exp(classifier_val[num_of_labels*(i*imagewidth+j)+l]) / sum);
                dataCost[num_of_labels*(i*imagewidth+j)+l] = data_cost_val;
            }
        }
    }


    delete []classifier_val;
    delete []classifier_val_reverse;

    // -----------------------------------------
    // end ALE boosted per pixel unary potential
    // -----------------------------------------


    // // reading cooc unary and pairwise terms
    // //char *filename2 = new char[200];
    // sprintf(filename, "eTRIMS/Result/Cooccurrence/cooccurence.dat");
    // int cooc_total = 0;
    // FILE *fp_in = fopen(filename, "r");
    // fscanf(fp_in, "%d", &cooc_total);

    // for(int i = 0; i < num_of_labels; i++)
    // {
    // 	float temp_val = 0;
    // 	fscanf(fp_in, "%f", &temp_val);
    // 	cooc_unary[i] = temp_val;
    // }

    // for(int i = 0; i < num_of_labels; i++)
    // 	for(int j = 0; j < num_of_labels; j++)
    // 	{
    // 		float temp_val = 0;
    // 		fscanf(fp_in, "%f", &temp_val);
    // 		cooc_pairwise[i*num_of_labels+j] = temp_val;
    // 	}

     //delete []filename;
}


void labelToRGB(short *map, const char *filename)
{
    for(int i1 = 0; i1 < imageheight; i1++) {
        for(int j1 = 0; j1 < imagewidth; j1++) {
            final_labels[(imageheight-(1+i1))*imagewidth+j1] = (unsigned char)(map[i1*imagewidth+j1])+1;
        }
    }

    LRgbImage rgbImage1(imagewidth, imageheight);
    unsigned char *rgbData = rgbImage1.GetData();

    for(int i1 = 0; i1 < imagewidth * imageheight; i1++, rgbData += 3) {
        dataset->LabelToRgb(&final_labels[i1], rgbData);
    }
    string finalFilename = string(dataset->testFolder) + filename + ".png";
    rgbImage1.Save(finalFilename.c_str());
}


void set_ho_layers()
{
    num_of_layers = 10;
    layers_dir = new char const*[num_of_layers];

    layers_dir[0] = "eTRIMS/Result/Segmentation/KMeans30/";
    layers_dir[1] = "eTRIMS/Result/Segmentation/KMeans40/";
    layers_dir[2] = "eTRIMS/Result/Segmentation/KMeans50/";
    layers_dir[3] = "eTRIMS/Result/Segmentation/KMeans60/";
    layers_dir[4] = "eTRIMS/Result/Segmentation/KMeans80/";
    layers_dir[5] = "eTRIMS/Result/Segmentation/KMeans100/";
    layers_dir[6] = "eTRIMS/Result/Segmentation/MeanShift70x70/";
    layers_dir[7] = "eTRIMS/Result/Segmentation/MeanShift70x100/";
    layers_dir[8] = "eTRIMS/Result/Segmentation/MeanShift100x70/";
    layers_dir[9] = "eTRIMS/Result/Segmentation/MeanShift100x100/";

    ho_stats_pot = "eTRIMS/Result/Stats/";
    ho_seg_ext = "seg";
    ho_sts_ext = "sts";
}


void set_det_layers()
{
    det_seg_dir = "eTRIMS/Result/Detectors/seg/";
    det_bb_dir  = "eTRIMS/Result/Detectors/bb/";
    det_seg_ext = "seg";
    det_bb_ext  = "bb";
}
