#pragma once
#include <vector>
#include <random>
#include <cassert>
#include "Tensor.h"
#include "LayerBase.h"
#include <immintrin.h>

inline double hsum_double_avx(__m256d v) {
	__m128d vlow = _mm256_castpd256_pd128(v);
	__m128d vhigh = _mm256_extractf128_pd(v, 1); // high 128
	vlow = _mm_add_pd(vlow, vhigh);     // reduce down to 128

	__m128d high64 = _mm_unpackhi_pd(vlow, vlow);
	return  _mm_cvtsd_f64(_mm_add_sd(vlow, high64));  // reduce to scalar
}

class ConvLayer : public LayerBase
{
public:
	ConvLayer(size_t filterWidth, size_t numFilters) : filterWidth(filterWidth), numFilters(numFilters) { type = Convolution; };
	size_t filterWidth;
	size_t numFilters;

	Tensor forward(Tensor& input, const Tensor* const W, const Tensor* const b)
	{
		Tensor result(input.gB(), input.gH() - filterWidth + 1u, input.gW() - filterWidth + 1u, numFilters);
		size_t resultB = result.gB();
		size_t resultH = result.gH();
		size_t resultW = result.gW();
		size_t resultC = result.gC();
		size_t inputB = input.gB();
		size_t inputH = input.gH();
		size_t inputW = input.gW();
		size_t inputC = input.gC();
		size_t WB = (*W).gB();
		size_t WH = (*W).gH();
		size_t WW = (*W).gW();
		size_t WC = (*W).gC();
		double* resultP = result.getEntriesPointer();
		double* WP = (*W).getEntriesPointer();
		double* inputP = input.getEntriesPointer();
		for (size_t bdx = 0; bdx < resultB; bdx++)
		{
			size_t resultBOffset = bdx * resultH * resultW * resultC;
			size_t inputBOffset = bdx * inputH * inputW * inputC;
			for (size_t hdx = 0; hdx < resultH; hdx++)
			{
				size_t resultHOffset = hdx * resultW * resultC;
				for (size_t wdx = 0; wdx < resultW; wdx++)
				{
					size_t resultWOffset = wdx * resultC;
					for (size_t fdx = 0; fdx < numFilters; fdx++)
					{
						size_t WBOffset = fdx * WH * WW * WC;
						size_t resultTotalOffset = resultBOffset + resultHOffset + resultWOffset + fdx;
						double bias = (*b)(fdx, 0, 0, 0);
						//result(bdx, hdx, wdx, fdx) = bias;
						result(resultTotalOffset) = bias;
						for (size_t fhdx = 0; fhdx < filterWidth; fhdx++)
						{
							size_t WHOffset = fhdx * WW * WC;
							size_t inputHOffset = (hdx + fhdx) * inputW * inputC;
							for (size_t fwdx = 0; fwdx < filterWidth; fwdx++)
							{
								size_t WWOffset = fwdx * WC;
								size_t inputWOffset = (wdx + fwdx) * inputC;
								/*
								for (size_t cdx = 0; cdx < inputC; cdx++)
								{
									//double value = input(bdx, hdx + fhdx, wdx + fwdx, cdx);
									double value = input(inputBOffset + inputHOffset + inputWOffset + cdx);
									//result(bdx, hdx, wdx, fdx) += value * (*W)(fdx, fhdx, fwdx, cdx);
									result(resultTotalOffset) += value * (*W)(WBOffset + WHOffset + WWOffset + cdx);
								}
								*/


								size_t inputFinalOffset = inputBOffset + inputHOffset + inputWOffset;
								size_t WFinalOffset = WBOffset + WHOffset + WWOffset;
								size_t cdx = 0;
								for (; cdx < (inputC & ~0x3); cdx += 4)
								{
									double value = inputP[inputFinalOffset + cdx];
									const __m256d valueAVX = _mm256_load_pd(&inputP[inputFinalOffset + cdx]);
									const __m256d WAVX = _mm256_load_pd(&WP[WFinalOffset + cdx]);
									const __m256d valueWAVX = _mm256_mul_pd(valueAVX, WAVX);

									resultP[resultTotalOffset] += hsum_double_avx(valueWAVX);
								}
								for (; cdx < inputC; cdx++)
								{
									double value = inputP[inputFinalOffset + cdx];
									resultP[resultTotalOffset] += value * WP[WFinalOffset + cdx];
								}
							}
						}
					}
				}
			}
		}
		return result;
	}

	Tensor backward(const Tensor& output, const Tensor* const input, const Tensor* const W, const Tensor* const b, Tensor* Wgrad, Tensor* bgrad)
	{
		Tensor result(output.gB(), output.gH() + filterWidth - 1u, output.gW() + filterWidth - 1u, input->gC());
		size_t outputB = output.gB();
		size_t outputH = output.gH();
		size_t outputW = output.gW();
		size_t outputC = output.gC();
		size_t resultB = result.gB();
		size_t resultH = result.gH();
		size_t resultW = result.gW();
		size_t resultC = result.gC();
		size_t WB = (*W).gB();
		size_t WH = (*W).gH();
		size_t WW = (*W).gW();
		size_t WC = (*W).gC();
		size_t inputB = (*input).gB();
		size_t inputH = (*input).gH();
		size_t inputW = (*input).gW();
		size_t inputC = (*input).gC();
		double* resultP = result.getEntriesPointer();
		double* WP = (*W).getEntriesPointer();
		double* inputP = (*input).getEntriesPointer();
		double* WgradP = (*Wgrad).getEntriesPointer();
		for (size_t bdx = 0; bdx < outputB; bdx++)
		{
			size_t resultBOffset = bdx * resultH * resultW * resultC;
			for (size_t hdx = 0; hdx < outputH; hdx++)
			{
				for (size_t wdx = 0; wdx < outputW; wdx++)
				{
					for (size_t fdx = 0; fdx < numFilters; fdx++)
					{
						size_t WBoffset = fdx * WH * WW * WC;
						double z = output(bdx, hdx, wdx, fdx);
						const __m256d zAVX = _mm256_set1_pd(z);
						(*bgrad)(fdx,0,0,0) += z;
						for (size_t fhdx = 0; fhdx < filterWidth; fhdx++)
						{
							size_t WHoffset = fhdx * WW * WC;
							size_t resultHOffset = (hdx + fhdx) * resultW * resultC;
							for (size_t fwdx = 0; fwdx < filterWidth; fwdx++)
							{
								size_t WWoffset = fwdx * WC;
								size_t WFinalOffset = WBoffset + WHoffset + WWoffset;
								size_t resultWOffset = (wdx + fwdx) * resultC;
								size_t resultFinalOffset = resultBOffset + resultHOffset + resultWOffset;
								
								/*
								for (size_t cdx = 0; cdx < inputC; cdx++)
								{
									resultP[resultFinalOffset + cdx] += z * WP[WFinalOffset + cdx];
									//result(resultBOffset + resultHOffset + resultWOffset + cdx) += z * (*W)(WBoffset + WHoffset + WWoffset + cdx);
									//result(bdx, hdx+fhdx, wdx + fwdx, cdx) += z * (*W)(fdx, fhdx, fwdx, cdx);

									//double inp = (*input)(bdx, hdx + fhdx, wdx + fwdx, cdx);
									double inp = inputP[resultFinalOffset + cdx];
									//double inp = (*input)(resultBOffset + resultHOffset + resultWOffset + cdx);
									//(*Wgrad)(fdx, fhdx, fwdx, cdx) += z * inp;
									WgradP[WFinalOffset + cdx] += z * inp;
									//(*Wgrad)(WBoffset + WHoffset + WWoffset + cdx) += z * inp;

								}
								*/
								size_t cdx = 0;
								for (; cdx < (inputC & ~0x3); cdx += 4)
								{
									const __m256d resultAVX = _mm256_load_pd(&resultP[resultFinalOffset + cdx]);
									const __m256d WAVX = _mm256_load_pd(&WP[WFinalOffset + cdx]);
									const __m256d zWAVX = _mm256_mul_pd(zAVX, WAVX);
									const __m256d tmp = _mm256_add_pd(zWAVX, resultAVX);
									_mm256_store_pd(&resultP[resultFinalOffset + cdx], tmp);

									const __m256d inputAVX = _mm256_load_pd(&inputP[resultFinalOffset + cdx]);
									const __m256d WgradAVX = _mm256_load_pd(&WgradP[WFinalOffset + cdx]);
									const __m256d zInputAVX = _mm256_mul_pd(zAVX, inputAVX);
									const __m256d tmp2 = _mm256_add_pd(zInputAVX, WgradAVX);
									_mm256_store_pd(&WgradP[WFinalOffset + cdx], tmp2);
								}
								for (; cdx < inputC; cdx++)
								{
									resultP[resultFinalOffset + cdx] += z * WP[WFinalOffset + cdx];
									double inp = inputP[resultFinalOffset + cdx];
									WgradP[WFinalOffset + cdx] += z * inp;
								}
								
							}
						}
					}
				}
			}
		}
		return result;
	}

};

