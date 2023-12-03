#pragma once
#include <vector>
#include <random>
#include <cassert>
#include "Tensor.h"
#include "LayerBase.h"
class SqError
{
public:
	SqError() { };

	Tensor backward(Tensor* input, const Tensor& expected)
	{
		Tensor result((*input).gB(),
			(*input).gH(),
			(*input).gW(),
			(*input).gC());

		for (size_t batchIndex = 0; batchIndex < (*input).gB(); batchIndex++)
		{
			for (size_t rowIndex = 0; rowIndex < result.gH(); rowIndex++)
			{
				for (size_t columnIndex = 0; columnIndex < result.gW(); columnIndex++)
				{
					for (size_t channelIndex = 0; channelIndex < (*input).gC(); channelIndex++)
					{
						double inputValue = (*input)(batchIndex,
							rowIndex,
							columnIndex,
							channelIndex);
						double outputValue = expected(batchIndex,
							rowIndex,
							columnIndex,
							channelIndex);
						result(batchIndex, rowIndex, columnIndex, channelIndex) += (-2.0 * outputValue) + (2.0 * inputValue);
					}
				}
			}
		}

		return result;
	}

	double forward(const Tensor& input, const Tensor& expectedOutput)
	{
		double result = 0;

		for (size_t batchIndex = 0; batchIndex < input.gB(); batchIndex++)
		{
			for (size_t rowIndex = 0; rowIndex < input.gH(); rowIndex++)
			{
				for (size_t columnIndex = 0; columnIndex < input.gW(); columnIndex++)
				{
					for (size_t channelIndex = 0; channelIndex < input.gC(); channelIndex++)
					{
						double inputValue = input(batchIndex,
							rowIndex,
							columnIndex,
							channelIndex);
						double outputValue = expectedOutput(batchIndex,
							rowIndex,
							columnIndex,
							channelIndex);
						result += (outputValue - inputValue) * (outputValue - inputValue);
					}
				}
			}
		}
		return result;
	}

private:

};

