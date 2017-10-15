#include "pdf.h"

// ----------------------------------------------------------------------------
// 1D distribution
// ----------------------------------------------------------------------------

DiscretePdf::DiscretePdf() : is_zero(true) {
}

DiscretePdf::DiscretePdf(const vector<float> &arr) {
    pdf.assign(arr.begin(), arr.end());
    normalize();
}

void DiscretePdf::normalize() {
    is_zero = false;

    if (pdf.size() == (vector<float>::size_type)0) {
        is_zero = true;
        return;
    }

    // normalize to create a probability distribution function
	float sum = 0.f;
	for (vector<float>::size_type i = 0; i < pdf.size(); ++i)
		sum += pdf[i];

    if (sum == 0.0f) {
        // not sampleable
        for (vector<float>::size_type i = 0; i < pdf.size(); ++i) {
		    pdf[i] = 0.0f;
        }
        is_zero = true;

    } else {
        float inv_sum = 1.0f / sum;
	    for (vector<float>::size_type i = 0; i < pdf.size(); ++i)
		    pdf[i] *= inv_sum;
    }
}

bool DiscretePdf::isValid() const {
    return ! is_zero;
}

DiscretePdf& DiscretePdf::operator*(const DiscretePdf &p) {
    if (this->size() != p.size())
        return *this;

    for (int i = 0; i < this->size(); ++i) {
        pdf[i] *= p[i];
    }

    normalize();

    return *this;
}

DiscretePdf& DiscretePdf::operator+(const DiscretePdf &p) {
    if (this->size() != p.size())
        return *this;

    for (int i = 0; i < this->size(); ++i) {
        pdf[i] += p[i];
    }

    normalize();

    return *this;
}
