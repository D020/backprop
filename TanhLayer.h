#pragma once
#include <vector>
#include <random>
#include <cassert>
#include "Tensor.h"
#include "LayerBase.h"
class TanhLayer : public LayerBase
{
public:
	TanhLayer()
	{
		type = Tanh;
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
					for (size_t channelIndex = 0; channelIndex < result.gC(); channelIndex++)
					{
						double z = output(batchIndex, rowIndex, columnIndex, channelIndex);
						double inputValue = (*input)(batchIndex, rowIndex, columnIndex, channelIndex);
						result(batchIndex, rowIndex, columnIndex, channelIndex) += z * (1 - tanh(inputValue) * tanh(inputValue));
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
							+= tanh(value);
					}
				}
			}
		}
		return result;
	}
};

