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

#include <iostream>
#include <sstream>
#include <fstream>
#include <iomanip>    // std::fixed, std::setprecision
#include <algorithm>  // std::fill
#include "evaluate.h"
#include "image.h"    // ale::LLabelImage

using namespace std;
using namespace ale;


void evaluateGroundTruth(LDataset* dataset, LList<char *>& imageFiles) {
    evaluateGroundTruth(dataset, imageFiles, string(dataset->testFolder) + "denseho.txt");
}


void evaluateGroundTruth(LDataset* dataset, LList<char *>& imageFiles, const string& output) {
    int* pixTotalClass = new int[dataset->classNo];
    int* pixOkClass    = new int[dataset->classNo];
    int* pixLabel      = new int[dataset->classNo];
    int* confusion     = new int[dataset->classNo * dataset->classNo];

    memset(pixTotalClass, 0, dataset->classNo * sizeof(int));
    memset(pixOkClass,    0, dataset->classNo * sizeof(int));
    memset(pixLabel,      0, dataset->classNo * sizeof(int));
    memset(confusion,     0, dataset->classNo * dataset->classNo * sizeof(int));

    unsigned int pixTotal = 0;
    unsigned int pixOk = 0;

    for (int i = 0; i < imageFiles.GetCount(); i++) {
        char *labeledImage = GetFileName(dataset->testFolder, imageFiles[i], ".png");
        char *gtImage      = GetFileName(dataset->groundTruthFolder, imageFiles[i], dataset->groundTruthExtension);

        cout << labeledImage << " " << gtImage << endl;

        LLabelImage labels(labeledImage, dataset, (void (LDataset::*)(unsigned char *, unsigned char *))&LDataset::RgbToLabel);
        LLabelImage groundTruth(gtImage, dataset, (void (LDataset::*)(unsigned char *, unsigned char *))&LDataset::RgbToLabel);

        unsigned char *labelData = labels.GetData();
        unsigned char *gtData = groundTruth.GetData();
        int points = groundTruth.GetPoints();

        for (int l = 0; l < points; l++, gtData++, labelData++) {
            if (*gtData != 0) {
                pixTotal++;
                pixTotalClass[*gtData - 1]++;
                pixLabel[*labelData - 1]++;
                if (*gtData == *labelData) {
                    pixOk++, pixOkClass[*gtData - 1]++;
                }
                confusion[(*gtData - 1) * dataset->classNo + *labelData - 1]++;
            }
        }
        delete[] labeledImage;
        delete[] gtImage;
    }
    double average = 0.0;
    double iou     = 0.0;  // intersection over union

    for (int i = 0; i < dataset->classNo; i++) {
        average += (pixTotalClass[i] == 0) ? 0 : pixOkClass[i] / (double) pixTotalClass[i];
        iou += (pixTotalClass[i] + pixLabel[i] - pixOkClass[i] == 0) ? 0 : pixOkClass[i] / (double)(pixTotalClass[i] + pixLabel[i] - pixOkClass[i]);
    }

    average /= dataset->classNo;
    iou     /= dataset->classNo;

    fstream ff(output, fstream::out);

    for (int i = 0 ; i < dataset->classNo; i++) {
        for (int w = 0; w < dataset->classNo; w++) {
            double value = 0;

            if (pixTotalClass[i] != 0) {
                value = confusion[i * dataset->classNo + w] / (double) pixTotalClass[i];
            }
            ff << fixed << setprecision(3) << value;
        }
        ff << "\n";
    }
    ff << "\n";

    for (int i = 0 ; i < dataset->classNo; i++) {
        double value = 0;

        if (pixTotalClass[i] + pixLabel[i] - pixOkClass[i] != 0) {
            value = pixOkClass[i] / (double) (pixTotalClass[i] + pixLabel[i] - pixOkClass[i]);
        }
        ff << fixed << setprecision(3) << value;
    }
    ff << "\n";

    for (int i = 0 ; i < dataset->classNo; i++) {
        double value = 0;

        if (pixTotalClass[i] != 0) {
            value = pixOkClass[i] / (double) pixTotalClass[i];
        }
        ff << fixed << setprecision(3) << value;
    }
    ff << "\n";

    double overall = (pixTotal != 0) ? pixOk / (double) pixTotal : 0.0;

    ff << "overall " << fixed << setprecision(4) << overall << ", "
       << "average " << fixed << setprecision(4) << average << ", "
       << "iou "     << fixed << setprecision(4) << iou << "\n";

    delete[] pixTotalClass;
    delete[] pixLabel;
    delete[] pixOkClass;
    delete[] confusion;
}
