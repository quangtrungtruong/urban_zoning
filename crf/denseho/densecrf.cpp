
// #pragma warning(disable : 4305)

#include "densecrf.h"
#include "fastmath.h"
#include "../../util.h"
#include <limits>
#include <iostream>

/////////////////////////////
/////  Alloc / Dealloc  /////
/////////////////////////////
DenseCRF::DenseCRF(int N, int M) : N_(N), M_(M) {

	// initialize higher order terms
	addHO = 0;
	addDet = 0;
	addCooc = 0;

	unary_ = allocate( N_*M_ );
	additional_unary_ = allocate( N_*M_ );
	current_ = allocate( N_*M_ );
	next_ = allocate( N_*M_ );
	tmp_ = allocate( 2*N_*M_ );
	memset( additional_unary_, 0, sizeof(float)*N_*M_ );
}
DenseCRF::~DenseCRF()
{
	deallocate( unary_ );
	deallocate( additional_unary_ );
	deallocate( current_ );
	deallocate( next_ );
	deallocate( tmp_ );
	for( unsigned int i=0; i<pairwise_.size(); i++ )
		delete pairwise_[i];
}

DenseCRF2D::DenseCRF2D(int W, int H, int M) : DenseCRF(W*H,M), W_(W), H_(H)
{
}

DenseCRF2D::~DenseCRF2D()
{
}
/////////////////////////////////
/////  Pairwise Potentials  /////
/////////////////////////////////
void DenseCRF::addPairwiseEnergy (const float* features, int D, float w, const SemiMetricFunction * function)
{
	if (function)
		addPairwiseEnergy( new SemiMetricPotential( features, D, N_, w, function ) );
	else
		addPairwiseEnergy( new PottsPotential( features, D, N_, w ) );
}

void DenseCRF::addPairwiseEnergy (float param, const float* features, int D, float w, const SemiMetricFunction * function)
{
    if (function)
        addPairwiseEnergy( new SemiMetricPotential( features, D, N_, w, function ) );
    else
        addPairwiseEnergy( new PottsPotential( features, D, N_, w ) );
}

void DenseCRF::addPairwiseEnergy(float param)
{
	addPairwiseEnergy(new CustomizePotential(N_, param));
}

void DenseCRF::addPairwiseEnergy ( PairwisePotential* potential )
{
	pairwise_.push_back( potential );
}

void DenseCRF2D::addPairwiseGaussian ( float sx, float sy, float w, const SemiMetricFunction * function )
{
	float * feature = new float [N_*2];
	for( int j=0; j<H_; j++ )
		for( int i=0; i<W_; i++ ){
			feature[(j*W_+i)*2+0] = i / sx;
			feature[(j*W_+i)*2+1] = j / sy;
		}
	addPairwiseEnergy( feature, 2, w, function );
	delete [] feature;
}

void DenseCRF2D::addPairwiseBilateral ( float sx, float sy, float sr, float sg, float sb, const unsigned char* im, float w, const SemiMetricFunction * function ) {
	float * feature = new float [N_*5];
	for( int j=0; j<H_; j++ )
		for( int i=0; i<W_; i++ ){
			feature[(j*W_+i)*5+0] = i / sx;
			feature[(j*W_+i)*5+1] = j / sy;
			feature[(j*W_+i)*5+2] = im[(i+j*W_)*3+0] / sr;
			feature[(j*W_+i)*5+3] = im[(i+j*W_)*3+1] / sg;
			feature[(j*W_+i)*5+4] = im[(i+j*W_)*3+2] / sb;
		}
	addPairwiseEnergy( feature, 5, w, function );
	delete [] feature;
}
//////////////////////////////
/////  Unary Potentials  /////
//////////////////////////////
void DenseCRF::setUnaryEnergy ( const float* unary/*,  float *cooc_unary, float *cooc_pairwise*/)
{
	memcpy( unary_, unary, N_*M_*sizeof(float) );
}

void DenseCRF::setUnaryEnergy ( int n, const float* unary )
{
	memcpy( unary_+n*M_, unary, M_*sizeof(float) );
}
void DenseCRF2D::setUnaryEnergy ( int x, int y, const float* unary ) {
	memcpy( unary_+(x+y*W_)*M_, unary, M_*sizeof(float) );
}
///////////////////////
/////  Inference  /////
///////////////////////

void DenseCRF::map ( int n_iterations, short* result, float relax)
{
	// Run inference
	float * prob = runInference( n_iterations, relax );

	// Find the map
	for (int i=0; i<N_; i++ ) {
		const float * p = prob + i * M_;
		float mx = p[0];
		int imx = 0;
		for( int j=1; j<M_; j++) {
			if( mx < p[j] ) {
				mx = p[j];
				imx = j;
			}
		}
		result[i] = imx;
	}
}

float* DenseCRF::runInference( int n_iterations, float relax )
{
	startInference();

	for( int it=0; it<n_iterations; it++) {
		stepInference(relax);
	}

	return current_;
}
void DenseCRF::expAndNormalize ( float* out, const float* in, float scale, float relax )
{
	float *V = new float[ N_+10 ];

	for(int i = 0; i < N_; i++ ) {
		const float * b = in + i*M_;

		// Find the max and subtract it so that the exp doesn't explode
		float mx = scale*b[0];
		for( int j=1; j<M_; j++ ) {
			if( mx < scale*b[j] ) {
				mx = scale*b[j];
			}
		}
		float tt = 0;
		for( int j=0; j<M_; j++ ) {
			V[j] = fast_exp( scale*b[j]-mx );
			tt += V[j];
		}

		// Make it a probability
		for( int j=0; j<M_; j++ ) {
			V[j] /= tt;
		}

		float * a = out + i*M_;
		for( int j=0; j<M_; j++ ) {
			if (relax == 1) {
				a[j] = V[j];
			} else {
				a[j] = (1-relax)*a[j] + relax*V[j];
			}
		}
	}
	delete[] V;
}



void DenseCRF::expAndNormalizeCooc ( float *cooc_out, float *cooc_in, float scale, float relax )
{
	float *V_cooc = new float[ M_+10 ];
	for( int i=0; i<M_; i++ )
	{
		const float * b_cooc = cooc_in + i*2;
		// Find the max and subtract it so that the exp doesn't explode
		float mx_cooc = scale*b_cooc[0];
		for( int j=1; j < 2; j++ )
			if( mx_cooc < scale * b_cooc[j] )
				mx_cooc = scale*b_cooc[j];

		float tt = 0;
		for( int j=0; j<2; j++ )
		{
			V_cooc[j] = fast_exp( scale*b_cooc[j]-mx_cooc );
			tt += V_cooc[j];
		}
		// Make it a probability
		for( int j=0; j<2; j++ )
			V_cooc[j] /= tt;

		float * a_cooc = cooc_out + i*2;
		for( int j=0; j<2; j++ )
			a_cooc[j] = V_cooc[j];
	}

 	delete[] V_cooc;
}



///////////////////
/////  Debug  /////
///////////////////

void DenseCRF::unaryEnergy(const short* ass, float* result) {
	for( int i=0; i<N_; i++ )
		if ( 0 <= ass[i] && ass[i] < M_ )
			result[i] = unary_[ M_*i + ass[i] ];
		else
			result[i] = 0;
}
void DenseCRF::pairwiseEnergy(const short* ass, float* result, int term)
{
	float * current = allocate( N_*M_ );
	// Build the current belief [binary assignment]
	for( int i=0,k=0; i<N_; i++ )
		for( int j=0; j<M_; j++, k++ )
			current[k] = (ass[i] == j);

	for( int i=0; i<N_*M_; i++ )
		next_[i] = 0;
	if (term == -1)
		for( unsigned int i=0; i<pairwise_.size(); i++ )
			pairwise_[i]->apply( next_, current, tmp_, M_ );
	else
		pairwise_[ term ]->apply( next_, current, tmp_, M_ );
	for( int i=0; i<N_; i++ )
		if ( 0 <= ass[i] && ass[i] < M_ )
			result[i] =-next_[ i*M_ + ass[i] ];
		else
			result[i] = 0;
	deallocate( current );
}


void DenseCRF::startInference()
{
	if(addCooc)
	{
		int *total_num_labels = new int[M_];
		memset(total_num_labels, 0, M_);
		for(int i = 0; i < N_; i++)
		{
			int class_label = 0; float temp_unary_cost = unary_[i*M_];
			for(int j = 1; j < M_; j++)
			{
				if(temp_unary_cost < unary_[i*M_+j])
				{
					temp_unary_cost = unary_[i*M_+j];
					class_label = j;
				}
			}
			total_num_labels[class_label]++;
		}

		float pairwise_cooc = 0.0; // float p1, p2, p, p12;
		for(int i = 0; i < M_; i++)
		{
			if(total_num_labels[i] > 0)
			{
				next_cooc_[2*i+1] = total_num_labels[i];
				next_cooc_[2*i] = 1;
			}
			else
			{
				next_cooc_[2*i+1] = 1;
				next_cooc_[2*i] = 100;
			}
		}

		delete []total_num_labels;
	}
	// Initialize using the unary energies
	expAndNormalize( current_, unary_, -1 );

	if(addCooc)
	{
		expAndNormalizeCooc ( current_cooc_, next_cooc_) ;
	}
}

void DenseCRF::stepInference( float relax )
{
#ifdef SSE_DENSE_CRF
	__m128 * sse_next_ = (__m128*)next_;
	__m128 * sse_unary_ = (__m128*)unary_;
	__m128 * sse_additional_unary_ = (__m128*)additional_unary_;
#endif
	// Set the unary potential
#ifdef SSE_DENSE_CRF

	static const __m128 SIGNMASK = _mm_castsi128_ps(_mm_set1_epi32(0x80000000));

	__m128 val1, val2;

	for( int i=0; i<(N_*M_-1)/4+1; i++ )
	{
		val1 = _mm_xor_ps(sse_unary_[i], SIGNMASK); val2 = _mm_xor_ps(sse_additional_unary_[i], SIGNMASK);
		sse_next_[i] = _mm_add_ps(val1, val2);
	}
#else
	for( int i=0; i<N_*M_; i++ )
		next_[i] = -unary_[i] - additional_unary_[i];
#endif

	//start PN Potts
	if(addHO)
	{
		initHO();
		int num_of_layers = num_layers;
		for(int i = 0; i < num_of_layers; i++)
			calculateHOPotential(i);

		for(int i = 0; i < N_*M_; i++)
		{
			next_[i] = next_[i] - higher_order[i];
		}
	}

	//end PN Potts


	// start add co-occurrence terms
	if(addCooc)
	{
		int *higher_labels = new int[M_];

		for(int i = 0; i < M_; i++)
		{
			if(current_cooc_[2*i] < current_cooc_[2*i+1])
				higher_labels[i] = 1;
			else
				higher_labels[i] = 0;
		}

		float *temp_prob_mult = new float[2*M_];
		float mult_prob = 0.0, mult_prob1 = 0.0;

		for(int i = 0; i < M_; i++)
		{
			mult_prob = 0.0; mult_prob1 = 0.0;
			for(int j = 0; j < N_; j++)
			{
				mult_prob = mult_prob + (1.0-current_[j*M_+i]);
				mult_prob1 = mult_prob1 + current_[j*M_+i];
			}

			temp_prob_mult[2*i] = mult_prob;
			temp_prob_mult[2*i+1] = mult_prob1;

			if(temp_prob_mult[2*i] < 1e-4) temp_prob_mult[2*i] = 1e-4;
			if(temp_prob_mult[2*i+1] < 1e-4) temp_prob_mult[2*i+1] = 1e-4;
		}

		float pairwise_cooc = 0.0; float p1, p2, p, p12;
		for(int i = 0; i < M_; i++)
		{
			pairwise_cooc = 0.0;
			p1 = unary_cooc_[i];
			for(int j = 0; j < M_; j++)
			{
				p2 = unary_cooc_[j];
				p12 = pair_cooc_[i*M_+j];
				p = 1 - (1 - p12 / p2) * (1 - p12 / p1);
				if(p > 1) p = 1;
				if(p < 1e-6) p = 1e-6;

				if(i != j)
					pairwise_cooc = pairwise_cooc - (0.005*N_) * log(p) * current_cooc_[j*2+1];
			}
			next_cooc_[2*i+1] = -pairwise_cooc;
		}

		for(int i = 0; i < M_; i++)
		{
			next_cooc_[2*i] = -1.0*cooc_factor*(temp_prob_mult[2*i+1]);
		}


		float temp_cooc_factor = 1.0;

		for(int i = 0; i < N_; i++)
		{
			for(int j = 0; j < M_; j++)
			{
				mult_prob = 1.0;
				next_[i*M_+j] = next_[i*M_+j] - cooc_factor*current_cooc_[2*j];
			}
		}

		delete []temp_prob_mult;
		delete []higher_labels;
	}
	// end co-occurrence terms

	// start det
	if(addDet)
	{
		calculateDetHOPotential();

		for(int i = 0; i < N_*M_; i++)
		{
			next_[i] = next_[i] + det_higher_order[i];
		}
	}

	// end det

	// pairwise potentials
	for( unsigned int i=0; i<pairwise_.size(); i++)
		pairwise_[i]->apply( next_, current_, tmp_, M_);

	// end pairwise

	// Exponentiate and normalize
	expAndNormalize( current_, next_, 1.0, relax );
	if(addCooc)
	{
		expAndNormalizeCooc ( current_cooc_, next_cooc_) ;
	}
}
void DenseCRF::currentMap( short * result )
{
	// Find the map
	for( int i=0; i<N_; i++ ){
		const float * p = current_ + i*M_;
		// Find the max and subtract it so that the exp doesn't explode
		float mx = p[0];
		int imx = 0;
		for( int j=1; j<M_; j++ )
			if( mx < p[j] ){
				mx = p[j];
				imx = j;
			}
		result[i] = imx;
	}
}


// start co-occurrence

void DenseCRF::setHOCooc(int add_Cooc)
{
	addCooc = 0;
	if(add_Cooc)
	{
		addCooc = 1;

		unary_cooc_ = new float[M_];
		memset(unary_cooc_, 0, sizeof(float)*M_);
		pair_cooc_ = new float[M_*M_];
		memset(pair_cooc_, 0, sizeof(float)*M_*M_);
		current_cooc_ = allocate(2*M_);
		next_cooc_ = allocate(2*M_);
	}
}

void DenseCRF::setCooccurence(float *cooc_unary, float *cooc_pairwise,  float coocFactor)
{
	if(addCooc)
	{
		memcpy( unary_cooc_, cooc_unary, M_*sizeof(float) );
		for(int i = 0; i < M_; i++)
			unary_cooc_[i] = unary_cooc_[i];

		memcpy(pair_cooc_, cooc_pairwise, M_*M_*sizeof(float));

		for(int i = 0; i < M_*M_; i++)
			pair_cooc_[i] = pair_cooc_[i];

		cooc_factor = coocFactor;
	}
}

// end co-occurrence



// start adding higher order potential related stuffs

void DenseCRF::setHO(int add_ho)
{
	addHO = 0;
	if(add_ho)
		addHO = 1;
}

void DenseCRF::initMemoryHO(int layers, float hoParam1, float hoParam2)
{
	if (addHO) {
		num_layers = layers;
		higher_order = new float[N_ * M_];

		for(int i = 0; i < N_ * M_; i++)
			higher_order[i] = 0.0;

        baseSegment = new int[N_];
		baseSegmentIndexes = new int**[num_layers];
		segmentCount = new int[num_layers];
		baseSegmentCounts = new int*[num_layers];
		ho_param1 = hoParam1;
		ho_param2 = hoParam2;
	}
}

void DenseCRF::setSegment(int* segment, int layer)
{
	if(addHO) {
		segmentCount[layer] = 0;
		for(int i = 0; i < N_; ++i) {
            baseSegment[i] = segment[i];
			if(segment[i] + 1 > segmentCount[layer])
				segmentCount[layer] = segment[i] + 1;
        }
		baseSegmentCounts[layer] = new int[segmentCount[layer]];
		memset(baseSegmentCounts[layer], 0, segmentCount[layer] * sizeof(int));

        // count the size of every segments
		for(int i = 0; i < N_; ++i)
			baseSegmentCounts[layer][segment[i]]++;

		int sum = 0;
		for(int i = 0; i < segmentCount[layer]; ++i)
			sum += baseSegmentCounts[layer][i];
        printf("N: %d\n", sum);

		baseSegmentIndexes[layer] = new int*[segmentCount[layer]];
		for(int i = 0; i < segmentCount[layer]; ++i)
			baseSegmentIndexes[layer][i] = new int[baseSegmentCounts[layer][i]];

		segmentationIndex = new int[segmentCount[layer]];
		memset(segmentationIndex, 0, segmentCount[layer] * sizeof(int));

		for(int i = 0; i < N_; ++i) {
			baseSegmentIndexes[layer][segment[i]][segmentationIndex[segment[i]]] = i;
			segmentationIndex[segment[i]]++;
		}
		delete []segmentationIndex;
	}
}

void DenseCRF::initStatsPotential(int* baseLabel)
{
	if(addHO) {
		int total = 0;
		for(int i = 0; i < num_layers; i++) {
			total += segmentCount[i];
		}

        printf("R: %d\n", total);
		stats_potential = new double[total * M_];
		for(int i = 0; i < total * M_; i++)
			stats_potential[i] = 0.0;

        int start_loc = 0;
		for (int i = 0; i < num_layers; i++) {
			for (int j = 0; j < segmentCount[i]; ++j) {
                for (int k = 0; k < baseSegmentCounts[i][j]; ++k) {
                    int index = baseSegmentIndexes[i][j][k];
                    int label = baseLabel[index];
                    stats_potential[start_loc + label] += 1.0;
                }
                start_loc = start_loc + M_;
			}
		}

        double sum;
        start_loc = 0;
        for (int i = 0; i < num_layers; i++) {
			for (int j = 0; j < segmentCount[i]; ++j) {
                sum = 0.0;
                for (int k = 0; k < M_; k++) {
                    sum += stats_potential[start_loc + k];
                }
                for (int k = 0; k < M_; k++) {
                    stats_potential[start_loc + k] /= sum;
                }
                start_loc = start_loc + M_;
			}
		}
	}
}


void DenseCRF::calculateHOPotential(int layer)
{
	if(addHO) {
		h_norm = new double[segmentCount[layer] * M_];

		float norm_val = 0.0;
		int basesegmentcounts = 0;
		int curr_pix_label = 0, curr_pix_index; // int x, y;

		double higher_order_prob;

		for(int i = 0; i < segmentCount[layer]; i++)
			for(int j = 0; j < M_; j++)
				h_norm[i*M_+j] = 1.0;

		for(int i = 0; i < segmentCount[layer]; i++)
		{
			basesegmentcounts = baseSegmentCounts[layer][i];
			higher_order_prob = 1.0;
			for(int j = 0; j < M_; j++)
			{
				higher_order_prob = 1.0;
				for(int k = 0; k < basesegmentcounts; k++)
				{
					curr_pix_index = baseSegmentIndexes[layer][i][k];
					higher_order_prob = higher_order_prob * current_[curr_pix_index*M_+j];
				}
				h_norm[i*M_+j] = higher_order_prob;
			}
		}


		double alpha = 0.5, maxcost, weight, costdata = 0.0; int start_loc = 0;

		for(int i = 0; i < layer; i++)
			start_loc = start_loc + segmentCount[i];

		start_loc = start_loc * M_;

		for(int i = 0; i < segmentCount[layer]; i++)
		{
			basesegmentcounts = baseSegmentCounts[layer][i];

			weight = 0.3 * basesegmentcounts;
			maxcost = -weight * log(alpha);

			for(int j = 0; j < basesegmentcounts; j++)
			{
				curr_pix_index = baseSegmentIndexes[layer][i][j];
				for(int k = 0; k < M_; k++)
				{
					higher_order_prob = h_norm[i*M_+k]/(current_[curr_pix_index*M_+k]+0.0001);
					costdata = - weight * log(stats_potential[start_loc+k]);
					higher_order[curr_pix_index*M_+k] += (ho_param1*costdata - ho_param2*higher_order_prob);
				}
			}
			start_loc = start_loc+M_;
		}
		delete []h_norm;
	}
}

void DenseCRF::initHO()
{
	if(addHO) {
		for(int i = 0; i < N_ * M_; i++)
			higher_order[i] = 0.0;
	}
}


void DenseCRF::clearMemory()
{
	if(addHO)
	{
		delete []higher_order;
		for(int i = 0; i < num_layers; i++)
		{
			for(int j = 0; j < segmentCount[i]; j++)
			{
				delete []baseSegmentIndexes[i][j];
			}
			delete []baseSegmentIndexes[i];
		}
		delete []baseSegmentIndexes;

		for(int i = 0; i < num_layers; i++)
		{
			delete []baseSegmentCounts[i];
		}
		delete []baseSegmentCounts;
		delete []segmentCount;
		delete []stats_potential;
	}
	if(addCooc)
	{
		delete []unary_cooc_;
		delete []pair_cooc_;
	}
	if(addDet)
	{
		delete []det_higher_order;
		delete []det_h_norm;
	}
}

// end higher order potential related stuffs


// start det

void DenseCRF::setDetHO(int add_det)
{
	addDet = 0;
	if(add_det) addDet = 1;
}

void DenseCRF::initMemoryDetHO(float param, float detParam1, float detParam2)
{
	if(addDet)
	{
		det_higher_order = new float[N_*M_];
		for(int i = 0; i < N_*M_; i++) {
			det_higher_order[i] = 0.0;
		}
		det_param1 = detParam1;
		det_param2 = detParam2;
        param0= param;
	}
}

void DenseCRF::setDetSegments(const std::vector<region>& detections)
{
	if(addDet)
	{
        det_segmentCount = detections.size();
        detections_ = detections;

		det_resp = new double[det_segmentCount];
        for (int i = 0; i < detections.size(); ++i) {
            det_resp[i] = 0.8;
        }
		det_h_norm = new double[det_segmentCount*M_];
	}
}


void DenseCRF::calculateDetHOPotential()
{
	if(addDet)
	{
		float norm_val = 0.0;

		int segment_size = 0;
		int curr_pix_label = 0, curr_pix_index; //int x, y;

		double higher_order_prob;

		for(int i = 0; i < det_segmentCount; i++)
			for(int j = 0; j < M_; j++)
 				det_h_norm[i*M_+j] = 1.0;

		for(int i = 0; i < detections_.size(); i++)
		{
			segment_size = detections_[i].indices.size();
			higher_order_prob = 1.0;
			for(int j = 0; j < M_; j++)
			{
				higher_order_prob = 1.0;
				for(int k = 0; k < segment_size; k++)
				{
					curr_pix_index = detections_[i].indices[k];
					higher_order_prob = higher_order_prob * current_[curr_pix_index*M_+j];
				}
				det_h_norm[i*M_+j] = higher_order_prob;
			}
		}

		for(int i = 0; i < N_ * M_; i++) {
			det_higher_order[i] = 0.0;
		}

		double alpha = 0.5, maxcost, weight, costdata = 0.0;

        std::vector<float*> hists(detections_.size());
        for (int i = 0; i < detections_.size(); ++i) {
            segment_size = detections_[i].indices.size();
            hists[i] = new float[M_];
            memset(hists[i], 0, M_ * sizeof(float));
            for (int j = 0; j < segment_size; ++j) {
                curr_pix_index = detections_[i].indices[j];
                for (int k = 0; k < M_; ++k) {
                    hists[i][k] += current_[curr_pix_index * M_ + k] + 0.0001;
                }
            }
        }

		for(int i = 0; i < detections_.size(); i++) {
			segment_size = detections_[i].indices.size();

			weight = 0.3 * segment_size;
			maxcost = -weight * log(alpha);
			costdata = 6.0 * segment_size * (det_resp[i] + 1.2);

			if (costdata < 0) costdata = 0;

            float sum = 0.0f;
            for (int k = 0; k < M_; ++k) sum += hists[i][k];

			for (int j = 0; j < segment_size; j++) {
				curr_pix_index = detections_[i].indices[j];

				for (int k = 0; k < M_; k++) {
                    float consistency_prob = (hists[i][k] + current_[curr_pix_index * M_ + k] - 0.0001) / (sum - current_[curr_pix_index * M_ + k] - 0.0001);
					det_higher_order[curr_pix_index * M_ + k] -= param0*consistency_prob*log(consistency_prob);
                    //det_higher_order[curr_pix_index * M_ + k] += (1.0 / M_)*log(consistency_prob);

					/*higher_order_prob = det_h_norm[i * M_ + k] / (current_[curr_pix_index * M_ + k] + 0.0001);
					if (detections_[i].type == k) {
						det_higher_order[curr_pix_index * M_ + k] -= det_param1 * costdata - det_param2 * higher_order_prob;
					}*/
				}
			}
		}

        for (int i = 0; i < detections_.size(); ++i) {
            delete[] hists[i];
        }
	}
}

// end det