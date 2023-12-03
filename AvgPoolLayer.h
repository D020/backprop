#pragma once
#include <vector>
#include <random>
#include <cassert>
#include "Tensor.h"
#include "LayerBase.h"

class AvgPoolLayer : public LayerBase
{
public:
	AvgPoolLayer(size_t stride)
		: stride(stride)
	{

		type = AveragePool;
	};

	Tensor backward(const Tensor& output, const Tensor* const input, const Tensor* const W, const Tensor* const b, Tensor* Wgrad, Tensor* bgrad)
	{
		Tensor result(output.gB(),
			output.gH() * stride,
			output.gW() * stride,
			output.gC());

		size_t batchSize = output.gB();
		size_t rows = output.gH();
		size_t columns = output.gW();

		//batchIndex* (rows * columns * channels) + row * (columns * channels) + column * channels + channel;
		for (size_t batchIndex = 0; batchIndex < batchSize; batchIndex++)
		{
			for (size_t rowIndex = 0; rowIndex < rows; rowIndex++)
			{
				for (size_t columnIndex = 0; columnIndex < columns; columnIndex++)
				{
					for (size_t filterHeightIndex = 0; filterHeightIndex < stride; filterHeightIndex++)
					{
						for (size_t filterWidthIndex = 0; filterWidthIndex < stride; filterWidthIndex++)
						{
							for (size_t channelIndex = 0; channelIndex < input->gC(); channelIndex++)
							{
								double value = output(batchIndex, rowIndex, columnIndex, channelIndex);
								result(batchIndex, rowIndex * stride + filterHeightIndex, columnIndex * stride + filterWidthIndex, channelIndex) += value * (1.0 / double(stride * stride));
								//*(resultP + resultPBatchOffset + resultPRowOffset + resultPColOffset + filterIndex) += value * *(wP + wPFilterOffset + wPHeightOffset + wPWidthOffset + channelIndex);
							}
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
			input.gH() / stride,
			input.gW() / stride,
			input.gC());

		size_t batchSize = input.gB();
		size_t rows = input.gH() / stride;
		size_t columns = input.gW() / stride;

		for (size_t batchIndex = 0; batchIndex < batchSize; batchIndex++)
		{
			for (size_t rowIndex = 0; rowIndex < rows; rowIndex++)
			{
				for (size_t columnIndex = 0; columnIndex < columns; columnIndex++)
				{
					for (size_t filterHeightIndex = 0; filterHeightIndex < stride; filterHeightIndex++)
					{
						for (size_t filterWidthIndex = 0; filterWidthIndex < stride; filterWidthIndex++)
						{
							for (size_t channelIndex = 0; channelIndex < input.gC(); channelIndex++)
							{
								double value = input(batchIndex,
									rowIndex * stride + filterHeightIndex,
									columnIndex * stride + filterWidthIndex,
									channelIndex);
								result(batchIndex, rowIndex, columnIndex, channelIndex) += value * (1.0 / double(stride * stride));
							}
						}
					}
				}
			}
		}
		return result;
	}
	size_t stride;

};