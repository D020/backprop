#pragma once
#include <vector>
#include <random>
#include <cassert>
#include "Tensor.h"
#include "LayerBase.h"
class SoftLayer : public LayerBase
{
public:
	SoftLayer()
	{
		type = Softmax;
	};

	Tensor backward(const Tensor& output, const Tensor* const input, const Tensor* const W, const Tensor* const b, Tensor* Wgrad, Tensor* bgrad)
	{
		Tensor result(input->gB(),
			input->gH(),
			input->gW(),
			input->gC());

		for (size_t batchIndex = 0; batchIndex < result.gB(); batchIndex++)
		{
			for (size_t rowIndex = 0; rowIndex < result.gH(); rowIndex++)
			{
				for (size_t columnIndex = 0; columnIndex < result.gW(); columnIndex++)
				{
					for (size_t j = 0; j < result.gC(); j++)
					{
						for (size_t i = 0; i < input->gC(); i++)
						{
							
							if (i == j)
								result(batchIndex, rowIndex, columnIndex, j) += output(batchIndex, rowIndex, columnIndex, i) * (*input)(batchIndex, rowIndex, columnIndex, i) * (1 - (*input)(batchIndex, rowIndex, columnIndex, j));
							else
								result(batchIndex, rowIndex, columnIndex, j) += output(batchIndex, rowIndex, columnIndex, i) * (*input)(batchIndex, rowIndex, columnIndex, i) * (0 - (*input)(batchIndex, rowIndex, columnIndex, j));
							
							/*
							if (i == j)
								result(batchIndex, rowIndex, columnIndex, j) += output(batchIndex, rowIndex, columnIndex, i) * gyl(batchIndex, rowIndex, columnIndex, i) * (1 - gyl(batchIndex, rowIndex, columnIndex, j));
							else
								result(batchIndex, rowIndex, columnIndex, j) += output(batchIndex, rowIndex, columnIndex, i) * gyl(batchIndex, rowIndex, columnIndex, i) * (0 - gyl(batchIndex, rowIndex, columnIndex, j));
							*/
						}
					}
				}
			}
		}

		return result;
	}

	Tensor forward(Tensor& input, const Tensor* const W, const Tensor* const b)
	{
		Tensor result(input.gB(),
			input.gH(),
			input.gW(),
			input.gC());

		for (size_t batchIndex = 0; batchIndex < result.gB(); batchIndex++)
		{
			for (size_t rowIndex = 0; rowIndex < result.gH(); rowIndex++)
			{
				for (size_t columnIndex = 0; columnIndex < result.gW(); columnIndex++)
				{
					double sum = 0;
					double max = input.maxChannel(batchIndex, rowIndex, columnIndex);
					for (size_t channelIndex = 0; channelIndex < result.gC(); channelIndex++)
					{
						double value = input(batchIndex,
							rowIndex,
							columnIndex,
							channelIndex);
						sum += exp(value - max);
					}
					for (size_t channelIndex = 0; channelIndex < result.gC(); channelIndex++)
					{
						double value = input(batchIndex,
							rowIndex,
							columnIndex,
							channelIndex);
						result(batchIndex,
							rowIndex,
							columnIndex,
							channelIndex)
							= exp(value - max) / sum + 1.0e-8;
					}
				}
			}
		}
		return result;
	}

};

