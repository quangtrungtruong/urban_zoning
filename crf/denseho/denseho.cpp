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

#include <cstdio>
#include <iostream>
#include <algorithm>
#include <string>

#include "argtable3.h"
#include "densecrf.h"
#include "higherorder.h"
#include "evaluate.h"

using namespace std;
using namespace ale;


// global arg_xxx structs
struct arg_lit *helpOpt;
struct arg_str *datasetOpt;
struct arg_file *outOpt;
struct arg_end *endOpt;


static void printDatasets(ostream& stream) {
    stream << "Currently available datasets:" << endl
    << "  - eTRIMS"  << endl
    << "  - CMP"     << endl;
}


int main( int argc, char* argv[])
{
    // the global arg_xxx structs are initialised within the argtable
    void *argtable[] = {
        helpOpt    = arg_litn("h", "help", 0, 1, "display this helpOpt and exit"),
        outOpt     = arg_filen("o", "out", nullptr, 0, 1, "Output text file for the result"),
        datasetOpt = arg_strn(nullptr, nullptr, "dataset", 1, 1, "Dataset that should be evaluated"),
        endOpt     = arg_end(20),
    };

    char progname[] = "denseho";

    int nerrors;
    nerrors = arg_parse(argc,argv,argtable);

    // special case: '--help' takes precedence over error reporting
    if (helpOpt->count > 0) {
        cout << "Usage: " << progname << endl;
        arg_print_syntax(stdout, argtable, "\n\n");
        cout << "Automatic Labeling Environment (ALE)" << endl << endl;
        arg_print_glossary(stdout, argtable, "  %-25s %s\n");

        cout << endl;
        printDatasets(cout);

        return 0;
    }

    // If the parser returned any errors then display them and exit
    if (nerrors > 0) {
        // Display the error details contained in the arg_end struct
        arg_print_errors(stderr, endOpt, progname);
        cerr << "Try '" << progname << " --help' for more information" << endl;

        return 1;
    }

    string datasetName(*datasetOpt->sval);

    // make matching case insensitive
    transform(datasetName.begin(), datasetName.end(), datasetName.begin(), ::tolower);

    if (datasetName == "etrims") {
        dataset = new LeTRIMS8Dataset(true);
    } else if (datasetName == "cmp") {
        dataset = new LCMPDataset(true);
    } else {
        cerr << "Error: Unknwon dataset '" << *datasetOpt->sval << "'" << endl
        << endl;
        printDatasets(cerr);

        return 1;
    }

	num_of_labels = dataset->classNo;
    num_files = dataset->testImageFiles.GetCount();

    cout << "Number of files: " << num_files << endl;

	for (int files = 0; files < num_files; files++) {
		cout << "solving image " << files << ": " << dataset->testImageFiles[files] << endl;
		mem_init(dataset->testImageFiles[files]);
		
		short * map = new short[imagewidth*imageheight];
		DenseCRF2D crf_plane(imagewidth, imageheight, num_of_labels);

		// unary
		crf_plane.setUnaryEnergy(dataCost);
		
		// pairwise
		crf_plane.addPairwiseGaussian( 3, 3, 3 );
		crf_plane.addPairwiseBilateral( 50, 50, 15, 15, 15, im_orig, 5);
		
		int ho_on = 0;
		int ho_det = 0;
		int ho_cooc = 0;

		//// set PN potts ho_order
		// ho_on = 1;
		crf_plane.set_ho(ho_on);
		if(ho_on) {
			set_ho_layers(); 
			crf_plane.ho_mem_init(imagewidth, imageheight, layers_dir, num_of_layers, ho_stats_pot, ho_seg_ext, ho_sts_ext, 0.0006, 1.0); 
			crf_plane.readSegments(dataset->testImageFiles[files]);
		}
		
		////set ho_det
		// ho_det = 1;
		crf_plane.set_hodet(ho_det);
		if(ho_det) {
			set_det_layers(); 
			crf_plane.det_ho_mem_init(imagewidth, imageheight, det_seg_dir, det_bb_dir, det_seg_ext, det_bb_ext, 0.00005, 1.0);
			crf_plane.det_readSegmentIndex(dataset->testImageFiles[files]);
		}
		
		//// cooccurrence
		// ho_cooc = 1;
		crf_plane.set_hocooc(ho_cooc);
		if(ho_cooc) {
			crf_plane.setco_occurrence(cooc_unary, cooc_pairwise, 10.0);
		}
		
		// start inference 
		clock_t start = clock();
		crf_plane.map(5, map);
		clock_t end = clock();
		printf("time taken %f\n", (end - start) / (float)CLOCKS_PER_SEC);
		
		crf_plane.del_mem_higherorder(); 

		// save the output
		labelToRGB(map, dataset->testImageFiles[files]);

		del_meminit();
		delete[] map;
	}


    string output;

    if (outOpt->count == 0) {
        output = string(dataset->testFolder) + "denseho.txt";
    } else {
        output = *outOpt->filename;
    }

	evaluateGroundTruth(dataset, dataset->testImageFiles, output);

	cout << endl << endl << "finished with processing"
		 << endl << endl << "Results are stored in '" << output << "'" << endl;

	delete dataset;
}
