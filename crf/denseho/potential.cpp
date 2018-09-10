#include "potential.h"
#include "../../util.h"
#include <iostream>


PairwisePotential::~PairwisePotential() {};


SemiMetricFunction::~SemiMetricFunction() {};



PottsPotential::~PottsPotential()
{
    deallocate( norm_ );
}


PottsPotential::PottsPotential(const float* features, int D, int N, float w, bool per_pixel_normalization) : N_(N), w_(w) 
{
    lattice_.init( features, D, N );
    norm_ = allocate( N );
    for ( int i=0; i<N; i++ )
        norm_[i] = 1;
    // Compute the normalization factor
    lattice_.compute( norm_, norm_, 1 );
    if ( per_pixel_normalization ) {
        // use a per pixel normalization
        for ( int i=0; i<N; i++ )
            norm_[i] = 1.0 / (norm_[i]+1e-20);
    }
    else {
        float mean_norm = 0;
        for ( int i=0; i<N; i++ )
            mean_norm += norm_[i];
        mean_norm = N / mean_norm;
        // use a per pixel normalization
        for ( int i=0; i<N; i++ )
            norm_[i] = mean_norm;
    }
}

void PottsPotential::apply(float* out_values, const float* in_values, float* tmp, int value_size) const
{

    lattice_.compute( tmp, in_values, value_size );
    
    for ( int i=0,k=0; i<N_; i++ )
    {
        for ( int j=0; j<value_size; j++, k++ )
        {
            out_values[k] += w_*norm_[i]*tmp[k];
        }
    }
}



CustomizePotential::~CustomizePotential(){}

CustomizePotential::CustomizePotential(int N, float w, bool per_pixel_normalization) : N_(N)
{
	N_ = N;
	weight_ = w;
	norm_ = allocate(N);
	for (int i = 0; i<N; i++)
		norm_[i] = 1;
	if (per_pixel_normalization) {
		// use a per pixel normalization
		for (int i = 0; i<N; i++)
			norm_[i] = 1.0 / (norm_[i] + 1e-20);
	}
	else {
		float mean_norm = 0;
		for (int i = 0; i<N; i++)
			mean_norm += norm_[i];
		mean_norm = N / mean_norm;
		// use a per pixel normalization
		for (int i = 0; i<N; i++)
			norm_[i] = mean_norm;
	}
}

void CustomizePotential::apply(float* out_values, const float* in_values, float* tmp, int value_size) const
{
	// this code modifies it to be suitable with CVPR paper
	for (int i = 0, k = 0; i<N_; i++)
	{
		float max = in_values[i*value_size];
		int m_index = 0;
		out_values[i*value_size] += -weight_;
		for (int j = 1; j<value_size; j++, k++)
		{
			if (in_values[i*value_size + j]>max)
			{
				out_values[i*value_size + m_index] += 2 * weight_;
				max = in_values[i*value_size + j];
				m_index = j;
				out_values[i*value_size + j] += -weight_;
			}
			else
				out_values[i*value_size + j] += weight_;
		}
	}
}


SemiMetricPotential::SemiMetricPotential(const float* features, int D, int N, float w,
                                         const SemiMetricFunction* function,
                                         bool per_pixel_normalization) :
                                        PottsPotential(features, D, N, w, per_pixel_normalization),
    function_(function) {}


void SemiMetricPotential::apply(float* out_values, const float* in_values, float* tmp, int value_size) const
{
    lattice_.compute( tmp, in_values, value_size );

    // To the metric transform
    float * tmp2 = new float[value_size];
    for ( int i=0; i<N_; i++ ) {
        float * out = out_values + i*value_size;
        float * t1  = tmp  + i*value_size;
        function_->apply( tmp2, t1, value_size );
        for ( int j=0; j<value_size; j++ )
            out[j] -= w_*norm_[i]*tmp2[j];
    }
    delete[] tmp2;
}

