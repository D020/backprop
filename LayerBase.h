#pragma once
#include "Tensor.h"
enum layerType { Convolution, Tanh, AveragePool, Softmax , ReLU,  None};
class LayerBase
{
public:
	virtual Tensor backward(const Tensor& output, const Tensor* const input, const Tensor* const W, const Tensor* const b, Tensor* Wgrad, Tensor* bgrad) = 0;
	virtual Tensor forward(Tensor& input, const Tensor* const W, const Tensor* const b) = 0;
	LayerBase() {};
	layerType type = None;
};

