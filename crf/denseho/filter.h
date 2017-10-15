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


// This function defines a simplified interface to the permutohedral lattice
// We assume a filter standard deviation of 1
class Filter{
protected:
    int n1_, o1_, n2_, o2_;
    Permutohedral * permutohedral_;
    // Don't copy
    Filter( const Filter& filter ){}
public:
    // Use different source and target features
    Filter( const float * source_features, int N_source, const float * target_features, int N_target, int feature_dim );
    // Use the same source and target features
    Filter( const float * features, int N, int feature_dim );
    //
    ~Filter();
    // Filter a bunch of values
    void filter( const float * source, float * target, int value_size );
};
