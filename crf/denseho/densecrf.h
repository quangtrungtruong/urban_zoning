#pragma once
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

#include <vector>
#include <cstdlib>
#include "potential.h"
#include "../../proposal.h"


class DenseCRF
{
protected:
	friend class BipartiteDenseCRF;

	// Number of variables and labels
	int N_, M_;
	float *unary_, *additional_unary_, *current_, *next_, *tmp_;

	// start cooc

	float *unary_cooc_, *pair_cooc_, *current_cooc_, *next_cooc_, *tmp_cooc_;
	float cooc_factor;

	// end cooc


	// Store all pairwise potentials
	std::vector<PairwisePotential*> pairwise_;

	// Run inference and return the pointer to the result
	float* runInference( int n_iterations, /*float *un_normalized_val,*/ float relax);

	// Auxillary functions
	void expAndNormalize( float* out, const float* in, float scale = 1.0, float relax = 1.0 );
	void expAndNormalizeCooc( float *out_cooc_, float *in_cooc_, float scale = 1.0, float relax = 1.0 );

	// Don't copy this object, bad stuff will happen
	DenseCRF( DenseCRF & o ){}

public:
	// Create a dense CRF model of size N with M labels
	DenseCRF( int N, int M );
	virtual ~DenseCRF();
	// Add  a pairwise potential defined over some feature space
	// The potential will have the form:    w*exp(-0.5*|f_i - f_j|^2)
	// The kernel shape should be captured by transforming the
	// features before passing them into this function
	void addPairwiseEnergy( const float * features, int D, float w=1.0f, const SemiMetricFunction * function=NULL );
	void addPairwiseEnergy(float param, const float * features, int D, float w=1.0f, const SemiMetricFunction * function=NULL );
	void addPairwiseEnergy(float weight);

	// Add your own favorite pairwise potential (ownwership will be transfered to this class)
	void addPairwiseEnergy( PairwisePotential* potential );

	// Set the unary potential for all variables and labels (memory order is [x0l0 x0l1 x0l2 .. x1l0 x1l1 ...])
	// void setUnaryEnergy( const float * unary );
	void setUnaryEnergy( const float * unary/*, float *cooc_unary, float *cooc_pairwise*/);

	// Set the unary potential for a specific variable
	void setUnaryEnergy( int n, const float * unary );

	// Run inference and return the probabilities
	void inference( int n_iterations, float* result, float relax=1.0 );

	// Run MAP inference and return the map for each pixel
	//void map( int n_iterations, short int* result, float relax=1.0 );

	void map( int n_iterations, short int* result, /*float *pix_prob, float *un_normalized_val,*/ float relax=1.0 );



	// Step by step inference
	void startInference();
	void stepInference( /*float *un_normalized_val,*/ float relax = 1.0 );
	void currentMap( short * result );

	// start cooccurrence
	void setHOCooc(int);
	void setCooccurence(float *cooc_unary, float *cooc_pairwise, float coocFactor);
	char addCooc;
	// end cooccurrence

	// start adding higher order reltated stuffs
	void setHO(int);
	char addHO;
	float *higher_order;
    int *baseSegment;
	int *segmentIndex;
    int *segmentCount;
    int **baseSegmentCounts;
    int ***baseSegmentIndexes;
    int *segmentationIndex;
	double *stats_potential, *h_norm;
	float ho_param1, ho_param2;
	void calculateHOPotential(int layer);
	void setSegment(int* segment, int layer);
	void initStatsPotential(int* baseLabel);
	void initHO();
	void clearMemory();
	void initMemoryHO(int layers, float hoParam1, float hoParam2);
	int num_layers;
	// end higher order related stuffs

    // start multi-resolution related stuffs
    /* bool enabled_hire; */
    /* void setHiRe(bool enabled); */
    /* void calculateHiRePotential(); */
    /* void setSegment(int* segment); */
    /* void initStatsPotential(int* labels); */
    /* void initHiRe(); */
    /* void clearMemory(); */
    /* void initMemoryHiRe(); */
    // end multi-resolution related stuffs


	// det start
	// start adding higher order reltated stuffs
	void setDetHO(int);
	char addDet;
	float *det_higher_order, det_param1, det_param2, param0;
	double *det_h_norm, *det_resp;
	int *det_segmentIndex, det_segmentCount, *det_type;
    std::vector<region> detections_;
	void calculateDetHOPotential();
	void setDetSegments(const std::vector<region>& detections);
	void initMemoryDetHO(float, float, float);
	// det end

public: /* Debugging functions */
	// Compute the unary energy of an assignment
	void unaryEnergy( const short * ass, float * result );

	// Compute the pairwise energy of an assignment (half of each pairwise potential is added to each of it's endpoints)
	void pairwiseEnergy( const short * ass, float * result, int term=-1 );
};

class DenseCRF2D:public DenseCRF
{
protected:
	// Width, height of the 2d grid
	int W_, H_;
public:
	// Create a 2d dense CRF model of size W x H with M labels
	DenseCRF2D( int W, int H, int M );
	virtual ~DenseCRF2D();
	// Add a Gaussian pairwise potential with standard deviation sx and sy
	void addPairwiseGaussian( float sx, float sy, float w, const SemiMetricFunction * function=NULL );

	// Add a Bilateral pairwise potential with spacial standard deviations sx, sy and color standard deviations sr,sg,sb
	void addPairwiseBilateral( float sx, float sy, float sr, float sg, float sb, const unsigned char * im, float w, const SemiMetricFunction * function=NULL );

	// Set the unary potential for a specific variable
	void setUnaryEnergy( int x, int y, const float * unary );
	using DenseCRF::setUnaryEnergy;
};
