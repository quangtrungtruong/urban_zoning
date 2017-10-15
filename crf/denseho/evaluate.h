#pragma once

#include "dataset.h"  // ale::LDataset


/**
 * Calculates overall, average and IoU for the labled images
 * and their ground truth
 */
void evaluateGroundTruth(ale::LDataset* dataset, ale::LList<char *>& imageFiles,
                         const std::string& output);

/**
 * The same as above but using the default output file "denseho.txt" in the
 * test directory of the dataset
 */
void evaluateGroundTruth(ale::LDataset* dataset, ale::LList<char *>& imageFiles);
