#pragma once
#include <cassert>
#include <stdlib.h>
#include <cmath>
class Tensor
{
public:
	void save(FILE* fptr)
	{
		fwrite(e, sizeof(double), b * h * w * c, fptr);
	}
	void load(FILE* fptr)
	{
		fread(e, sizeof(double), b * h * w * c, fptr);
	}
	Tensor(size_t b, size_t h, size_t w, size_t c): b(b), h(h), w(w), c(c)
	{
		e = (double*) malloc(sizeof(double) * b * h * w * this->c);//new double[b*h*w*c]();
		if (!e)
			std::cout << "Tensor allocation failed." << std::endl;
		initZero();
	}
	Tensor() { b = 0; h = 0; w = 0; c = 0; e = NULL; };
	~Tensor() 
	{ 
		free(e);
	}
	Tensor(Tensor& other)   //copy constructor
	{
		b = other.b;
		h = other.h;
		w = other.w;
		c = other.c;

		e = (double*) malloc(sizeof(double) * b * h * w * c);
		if (!e)
			std::cout << "Tensor allocation failed." << std::endl;
		memcpy(e, other.e, sizeof(double) * b * h * w * c);

	}
	Tensor(const Tensor& other)   //copy constructor
	{
		b = other.b;
		h = other.h;
		w = other.w;
		c = other.c;

		e = (double*) malloc(sizeof(double) * b * h * w * c);
		if (!e)
			std::cout << "Tensor allocation failed." << std::endl;
		memcpy(e, other.e, sizeof(double) * b * h * w * c);

	}

	Tensor& operator=(const Tensor& other)
	{
		b = other.b;
		h = other.h;
		w = other.w;
		c = other.c;

		free(e);
		e = (double*)malloc(sizeof(double) * b * h * w * c);
		if (!e)
			std::cout << "Tensor allocation failed." << std::endl;
		memcpy(e, other.e, sizeof(double) * b * h * w * c);

		return *this;
	}

	Tensor operator+(const Tensor& other) const
	{
		assert(b == other.b);
		assert(h == other.h);
		assert(w == other.w);
		assert(c == other.c);
		Tensor r(b, h, w, c);
		for (size_t edx = 0; edx < b * h * w * c; edx++)
		{
			r.e[edx] = this->e[edx] + other.e[edx];
		}
		return r;
	}

	Tensor operator+(double add) const
	{
		Tensor r(b, h, w, c);
		for (size_t edx = 0; edx < b * h * w * c; edx++)
		{
			r.e[edx] = this->e[edx] + add;
		}
		return r;
	}

	Tensor operator-(const Tensor& other) const
	{
		assert(b == other.b);
		assert(h == other.h);
		assert(w == other.w);
		assert(c == other.c);
		Tensor r(b, h, w, c);
		for (size_t edx = 0; edx < b * h * w * c; edx++)
		{
			r.e[edx] = this->e[edx] - other.e[edx];
		}
		return r;
	}

	Tensor operator*(double scalar)
	{
		Tensor r(b, h, w, c);
		for (size_t edx = 0; edx < b * h * w * c; edx++)
		{
			r.e[edx] = this->e[edx] * scalar;
		}
		return r;
	}

	Tensor operator/(const Tensor& other) const
	{
		assert(b == other.b);
		assert(h == other.h);
		assert(w == other.w);
		assert(c == other.c);
		Tensor r(b, h, w, c);
		for (size_t edx = 0; edx < b * h * w * c; edx++)
		{
			r.e[edx] = this->e[edx] / other.e[edx];
		}
		return r;
	}

	Tensor sqrt()
	{
		Tensor r(b, h, w, c);
		for (size_t edx = 0; edx < b * h * w * c; edx++)
		{
			r.e[edx] = std::sqrt(this->e[edx]);
		}
		return r;
	}

	Tensor squared()
	{
		Tensor r(b, h, w, c);
		for (size_t edx = 0; edx < b * h * w * c; edx++)
		{
			r.e[edx] = this->e[edx] * this->e[edx];
		}
		return r;
	}

	void initRandom()
	{
		for (size_t edx = 0; edx < b * h * w * c; edx++)
			e[edx] = -1.0 + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (1.0 - -1.0)));
	}

	void initZero()
	{
		for (size_t edx = 0; edx < b * h * w * c; edx++)
			e[edx] = 0.0;
	}

	Tensor sliceBatch(size_t fromBatch, size_t toBatch, size_t* activeBatches) const
	{
		Tensor slice(toBatch - fromBatch, h, w, c);
		if (b < fromBatch)
			fromBatch = b;
		if (b < toBatch)
			toBatch = b;
		*activeBatches = 0;
		for (size_t bdx = fromBatch; bdx < toBatch; bdx++)
		{
			for (size_t hdx = 0; hdx < h; hdx++)
			{
				for (size_t wdx = 0; wdx < w; wdx++)
				{
					for (size_t cdx = 0; cdx < c; cdx++)
					{
						slice(bdx - fromBatch, hdx, wdx, cdx) = (*this)(bdx, hdx, wdx, cdx);
					}
				}
			}
			(*activeBatches)++;
		}
		return slice;
	}

	double& operator ()(size_t batchIndex, size_t row, size_t column, size_t channel)
	{
		assert(batchIndex < b);
		assert(row < h);
		assert(column < w);
		assert(channel < c);
		return e[batchIndex * (h * w * c) + row * (w * c) + column * c + channel];
	}

	double& operator()(size_t edx)
	{
		return e[edx];
	}

	double& operator()(size_t edx) const
	{
		return e[edx];
	}

	double* getEntriesPointer()
	{
		return e;
	}

	double* getEntriesPointer() const
	{
		return e;
	}

	const double& operator ()(size_t batchIndex, size_t row, size_t column, size_t channel) const
	{
		assert(batchIndex < b);
		assert(row < h);
		assert(column < w);
		assert(channel < c);
		return e[batchIndex * (h * w * c) + row * (w * c) + column * c + channel];
	}

	double maxChannel(size_t batchIndex, size_t row, size_t column) const
	{
		double max = (*this)(batchIndex, row, column, 0);
		for (size_t mdx = 0; mdx < c; mdx++)
		{
			double value = (*this)(batchIndex, row, column, mdx);
			if (max < value)
				max = value;
		}

		return max;
	}

	double max() const
	{
		double max = e[0];
		for (size_t mdx = 0; mdx < b * h * w * c; mdx++)
		{
			double value = e[mdx];
			if (value < max)
				continue;
			max = value;
		}

		return max;
	}

	bool checkNan() 
	{
		for (size_t edx = 0; edx < b * h * w * c; edx++)
			if (isnan(e[edx]))
				return true;
		return false;
	}

	bool checkInf()
	{
		for (size_t edx = 0; edx < b * h * w * c; edx++)
			if (isinf(e[edx]))
				return true;
		return false;
	}

	bool checkDenormal()
	{
		for (size_t edx = 0; edx < b * h * w * c; edx++)
			if (!isnormal(e[edx]))
				return true;
		return false;
	}

	size_t maxChannelIndex(size_t batchIndex, size_t row, size_t column) const
	{
		double max = (*this)(batchIndex, row, column, 0);
		size_t maxIndex = 0;
		for (size_t mdx = 0; mdx < c; mdx++)
		{
			double value = (*this)(batchIndex, row, column, mdx);
			if (value < max)
				continue;
			max = value;
			maxIndex = mdx;
		}

		return maxIndex;
	}

	size_t gB() const { return b; }
	size_t gH() const { return h; }
	size_t gW() const { return w; }
	size_t gC() const { return c; }

	void print()
	{
		for (size_t bdx = 0; bdx < b; bdx++)
		{
			for (size_t hdx = 0; hdx < h; hdx++)
			{
				for (size_t wdx = 0; wdx < w; wdx++)
				{
					for (size_t cdx = 0; cdx < c; cdx++)
					{
						std::cout << "[" << bdx << "][" << hdx << "][" << wdx << "][" << cdx << "] = " << (*this)(bdx, hdx, wdx, cdx) << std::endl;
					}
				}
			}
		}
	}

private:
	size_t b;
	size_t h;
	size_t w;
	size_t c;
	double* e;
};