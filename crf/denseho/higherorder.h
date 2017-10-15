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
#include <string>
#include "dataset.h"


extern ale::LDataset* dataset;

extern int num_of_labels;
extern int num_files;

extern float *dataCost;
extern float *cooc_unary;
extern float *cooc_pairwise;

extern int imagewidth;
extern int imageheight;
extern int K;

extern unsigned char *final_labels;
extern unsigned char *im_orig;

// ho_terms
extern int num_of_layers;
extern char const **layers_dir;
extern char const *ho_stats_pot;
extern char const *ho_seg_ext;
extern char const *ho_sts_ext;

// det_terms
extern char const *det_seg_dir;
extern char const *det_bb_dir;
extern char const *det_seg_ext;
extern char const *det_bb_ext;


void mem_init(const char *TrainFileName);
void del_meminit();

void loadFiles(const char *);
void labelToRGB(short *, const char *filename);

void set_ho_layers();
void set_det_layers();
