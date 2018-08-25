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

#include "permutohedral.h"


class PairwisePotential {
  public:
    virtual ~PairwisePotential();
    virtual void apply( float * out_values, const float * in_values, float * tmp, int value_size ) const = 0;
};


class SemiMetricFunction {
  public:
    virtual ~SemiMetricFunction();

    // For two probabilities apply the semi metric transform: v_i = sum_j mu_ij u_j
    virtual void apply( float * out_values, const float * in_values, int value_size ) const = 0;
};


class PottsPotential: public PairwisePotential
{
protected:
    Permutohedral lattice_;
    int N_;
    float w_;
    float *norm_;
public:
    PottsPotential(const float* features, int D, int N, float w, bool per_pixel_normalization = true);
    ~PottsPotential();

    void apply(float* out_values, const float* in_values, float* tmp, int value_size) const;
};

class CustomizePotential : public PairwisePotential
{
protected:
	float weight_;
	int N_;
	float *norm_;
public:
	CustomizePotential(int N, float w, bool per_pixel_normalization = true);
	~CustomizePotential();

	void apply(float* out_values, const float* in_values, float*tmp, int value_size) const;
};


class SemiMetricPotential: public PottsPotential
{
protected:
    const SemiMetricFunction * function_;
public:
    SemiMetricPotential(const float* features, int D, int N, float w, const SemiMetricFunction* function,
                        bool per_pixel_normalization = true);

    void apply(float* out_values, const float* in_values, float* tmp, int value_size) const;
};