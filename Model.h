#pragma once
#include "Tensor.h"
#include <vector>
#include <thread>
#include "LayerBase.h"
#include "ConvLayer.h"
#include "TanhLayer.h"
#include "AvgPoolLayer.h"
#include "SoftLayer.h"
#include "SqError.h"
#include "CrossError.h"
#include "ReLULayer.h"
#include <string>
class Model
{
public:
    Model(size_t cores, size_t batchSize) : cores(cores), batchSize(batchSize) { weightGradients.resize(cores); biasGradients.resize(cores); inputs.resize(cores); };

    Tensor predict(const Tensor& x, size_t core)
    {
        *(inputs[core][0]) = x;
        for (int ldx = 0; ldx < layers.size(); ldx++)
        {
            *(inputs[core][ldx+1]) = layers[ldx]->forward(*(inputs[core][ldx]), weights[ldx], biases[ldx]);
        }

        return *(inputs[core][layers.size()]);
    }

    
    void backward(const Tensor& expectedOutput, size_t core)
    {
        Tensor tmp = lossFunc->backward(inputs[core][layers.size()],expectedOutput);
        //std::cout << "Forward done." << std::endl;
        for (int ldx = layers.size() - 1; 0 <= ldx; ldx--)
        {
            if(layers[ldx]->type != Softmax)
                tmp = layers[ldx]->backward(tmp,inputs[core][ldx],weights[ldx],biases[ldx],weightGradients[core][ldx],biasGradients[core][ldx]);
            else
                tmp = layers[ldx]->backward(tmp, inputs[core][ldx+1], weights[ldx], biases[ldx], weightGradients[core][ldx], biasGradients[core][ldx]);
        }
    }

    void saveModel(std::string path)
    {
        FILE* fptr;
        fopen_s(&fptr, path.c_str(), "wb");
        for (int ldx = 0; ldx < layers.size(); ldx++)
        {
            weights[ldx]->save(fptr);
            biases[ldx]->save(fptr);
        }
        fclose(fptr);

    }
    
    void loadModel(std::string path)
    {
        FILE* fptr;
        fopen_s(&fptr, path.c_str(), "rb");
        for (int ldx = 0; ldx < layers.size(); ldx++)
        {
            weights[ldx]->load(fptr);
            biases[ldx]->load(fptr);
        }
        fclose(fptr);

    }

    void trainOnCore(const Tensor& input, const Tensor& output, size_t core)
    {
        for (int ldx = 0; ldx < layers.size(); ldx++)
        {
            weightGradients[core][ldx]->initZero();
            biasGradients[core][ldx]->initZero();
        }
        predict(input, core);
        backward(output, core);
    }

    size_t trainMulti(const Tensor& input, const Tensor& output, double learningRate, size_t timestep)
    {
        for (size_t batchdx = 0; batchdx < ceil(input.gB() / (batchSize * cores)); batchdx++)
        {
            std::vector<Tensor> inputSlices;
            std::vector<Tensor> outputSlices;
            std::vector<size_t> activeBatches;
            size_t offset = batchdx * batchSize * cores;            
            for (size_t core = 0; core < cores; core++)
            {
                size_t activeBatch = 0;
                inputSlices.push_back(input.sliceBatch(core * batchSize + offset, (core + 1) * batchSize + offset, &activeBatch));
                outputSlices.push_back(output.sliceBatch(core * batchSize + offset, (core + 1) * batchSize + offset, &activeBatch));
                activeBatches.push_back(activeBatch+1);
            }

            std::vector<std::thread> threads;
            for (int cdx = 0; cdx < cores; cdx++)
            {
                std::thread batchTrain(&Model::trainOnCore, this, inputSlices[cdx], outputSlices[cdx], cdx);
                threads.push_back(std::move(batchTrain));
            }

            for (int cdx = 0; cdx < cores; cdx++)
            {
                threads[cdx].join();
            }

            for (size_t ldx = 0; ldx < layers.size(); ldx++)
            {
                Gweights[ldx]->initZero();
            }

            for (size_t core = 0; core < cores; core++)
            {
                for (int ldx = 0; ldx < layers.size(); ldx++)
                {
                    *Gweights[ldx] = (*Gweights[ldx]) + (*(weightGradients[core][ldx]) * (1.0 / (double(cores) * double(activeBatches[core]))));
                    *Gbiases[ldx] = (*Gbiases[ldx]) + (*(biasGradients[core][ldx]) * (1.0 / (double(cores) * double(activeBatches[core]))));
                    //*weights[ldx] = (*weights[ldx]) - (*(weightGradients[core][ldx]) * (learningRate / (double(cores) * double(activeBatches[core]))));
                    //*biases[ldx] = (*biases[ldx]) - (*(biasGradients[core][ldx]) * (learningRate / (double(cores) * double(activeBatches[core]))));
                }
            }

            //m(t) = beta1 * m(t - 1) + (1 – beta1) * g(t)
            //v(t) = beta2 * v(t - 1) + (1 – beta2) * g(t) ^ 2
            for (int ldx = 0; ldx < layers.size(); ldx++)
            {
                *Mweights[ldx] = (*Mweights[ldx]) * beta1 + (*Gweights[ldx]) * (1 - beta1);
                *Mbiases[ldx] = (*Mbiases[ldx]) * beta1 + (*Gbiases[ldx]) * (1 - beta1);

                *Vweights[ldx] = (*Vweights[ldx]) * beta2 + (*Gweights[ldx]).squared() * (1 - beta2);
                *Vbiases[ldx] = (*Vbiases[ldx]) * beta2 + (*Gbiases[ldx]).squared() * (1 - beta2);

                Tensor Mhatweights = (*Mweights[ldx]) * (1.0 / (1 - std::pow(beta1, double(timestep+1))));
                Tensor Mhatbiases = (*Mbiases[ldx]) * (1.0 / (1 - std::pow(beta1, double(timestep + 1))));

                Tensor Vhatweights = (*Vweights[ldx]) * (1.0 / (1 - std::pow(beta2, double(timestep + 1))));
                Tensor Vhatbiases = (*Vbiases[ldx]) * (1.0 / (1 - std::pow(beta2, double(timestep + 1))));

                *weights[ldx] = (*weights[ldx]) - (Mhatweights / (Vhatweights.sqrt() + 0.0000001)) * learningRate;
                *biases[ldx] = (*biases[ldx]) - (Mhatbiases / (Vhatbiases.sqrt() + 0.0000001)) * learningRate;
            }
            timestep++;
        }
    }

    void train(const Tensor& input, const Tensor& output, double learningRate)
    {
        for (size_t core = 0; core < cores; core++)
        {
            for (int ldx = 0; ldx < layers.size(); ldx++)
            {
                weightGradients[core][ldx]->initZero();
                biasGradients[core][ldx]->initZero();
            }
        }
        for (size_t core = 0; core < cores; core++)
        {
            predict(input, core);
        }
        for (size_t core = 0; core < cores; core++)
        {
            backward(output, core);
        }
        
        for (size_t core = 0; core < cores; core++)
        {
            for (int ldx = 0; ldx < layers.size(); ldx++)
            {
                //weightGradients[core][ldx]->print();
                *weights[ldx] = (*weights[ldx]) - (*(weightGradients[core][ldx]) * (learningRate/(double(cores)*double(input.gB()))));
                *biases[ldx] = (*biases[ldx]) - (*(biasGradients[core][ldx]) * (learningRate / (double(cores) *double(input.gB()))));
            }
        }
    }
    
    double totalLossMulti(const Tensor& x, const Tensor& expectedOutput)
    {
        double totalLoss = 0;
        double* losses = (double*) malloc(sizeof(double) * cores);
        
        for (size_t batchdx = 0; batchdx < floor(x.gB() / (batchSize * cores)); batchdx++)
        {
            std::vector<Tensor> inputSlices;
            std::vector<Tensor> outputSlices;
            size_t offset = batchdx * batchSize * cores;
            size_t activeBatch = 0;
            for (size_t core = 0; core < cores; core++)
            {
                inputSlices.push_back(x.sliceBatch(core * batchSize + offset, (core + 1) * batchSize + offset, &activeBatch));
                outputSlices.push_back(expectedOutput.sliceBatch(core * batchSize + offset, (core + 1) * batchSize + offset, &activeBatch));
            }

            std::vector<std::thread> threads;
            for (int cdx = 0; cdx < cores; cdx++)
            {
                std::thread batchLoss(&Model::totalLoss, this, inputSlices[cdx], outputSlices[cdx], cdx, &losses[cdx]);
                threads.push_back(std::move(batchLoss));
            }

            for (int cdx = 0; cdx < cores; cdx++)
            {
                threads[cdx].join();
            }
            for (int cdx = 0; cdx < cores; cdx++)
            {
                totalLoss += losses[cdx];
            }

        }
        free(losses);
        return totalLoss/double(x.gB());
    }

    void totalLoss(const Tensor& x, const Tensor& expectedOutput, size_t core, double* loss)
    {
        Tensor predictions = predict(x, core);
        double result = lossFunc->forward(predictions, expectedOutput);
        *loss = result;
    }
    

    void addConvLayer(size_t filterWidth, size_t inputChannels, size_t numberOfFilters, size_t height, size_t width)
    {
        Tensor* W = new Tensor(numberOfFilters, filterWidth, filterWidth, inputChannels);
        Tensor* GW = new Tensor(numberOfFilters, filterWidth, filterWidth, inputChannels);
        Tensor* MW = new Tensor(numberOfFilters, filterWidth, filterWidth, inputChannels);
        Tensor* VW = new Tensor(numberOfFilters, filterWidth, filterWidth, inputChannels);
        Tensor* b = new Tensor(numberOfFilters, 1, 1, 1);
        Tensor* Gb = new Tensor(numberOfFilters, 1, 1, 1);
        Tensor* Mb = new Tensor(numberOfFilters, 1, 1, 1);
        Tensor* Vb = new Tensor(numberOfFilters, 1, 1, 1);

        W->initRandom();
        GW->initZero();
        MW->initZero();
        VW->initZero();
        b->initRandom();
        Gb->initZero();
        Mb->initZero();
        Vb->initZero();


        weights.push_back(W);
        Gweights.push_back(GW);
        Mweights.push_back(MW);
        Vweights.push_back(VW);
        biases.push_back(b);
        Gbiases.push_back(Gb);
        Mbiases.push_back(Mb);
        Vbiases.push_back(Vb);

        for (size_t core = 0; core < cores; core++)
        {
            Tensor* lossOverW = new Tensor(numberOfFilters, filterWidth, filterWidth, inputChannels);
            lossOverW->initZero();
            weightGradients[core].push_back(lossOverW);
            Tensor* lossOverb = new Tensor(numberOfFilters, 1, 1, 1);
            lossOverb->initZero();
            biasGradients[core].push_back(lossOverb);
            Tensor* input = new Tensor(batchSize,height,width, inputChannels);
            inputs[core].push_back(input);
        }

        ConvLayer* layer = new ConvLayer(filterWidth, numberOfFilters);
        layers.push_back(layer);

    }

    void addTanhLayer(size_t height, size_t width, size_t inputChannels)
    {
        Tensor* W = new Tensor();
        Tensor* GW = new Tensor();
        Tensor* MW = new Tensor();
        Tensor* VW = new Tensor();
        Tensor* b = new Tensor();
        Tensor* Gb = new Tensor();
        Tensor* Mb = new Tensor();
        Tensor* Vb = new Tensor();

        weights.push_back(W);
        Gweights.push_back(GW);
        Mweights.push_back(MW);
        Vweights.push_back(VW);
        biases.push_back(b);
        Gbiases.push_back(Gb);
        Mbiases.push_back(Mb);
        Vbiases.push_back(Vb);

        for (size_t core = 0; core < cores; core++)
        {
            Tensor* lossOverW = new Tensor();
            weightGradients[core].push_back(lossOverW);
            Tensor* lossOverb = new Tensor();
            biasGradients[core].push_back(lossOverb);
            Tensor* input = new Tensor(batchSize, height, width, inputChannels);
            inputs[core].push_back(input);
        }

        TanhLayer* layer = new TanhLayer();
        layers.push_back(layer);
    }

    void addReLULayer(size_t height, size_t width, size_t inputChannels)
    {
        Tensor* W = new Tensor();
        Tensor* GW = new Tensor();
        Tensor* MW = new Tensor();
        Tensor* VW = new Tensor();
        Tensor* b = new Tensor();
        Tensor* Gb = new Tensor();
        Tensor* Mb = new Tensor();
        Tensor* Vb = new Tensor();

        weights.push_back(W);
        Gweights.push_back(GW);
        Mweights.push_back(MW);
        Vweights.push_back(VW);
        biases.push_back(b);
        Gbiases.push_back(Gb);
        Mbiases.push_back(Mb);
        Vbiases.push_back(Vb);

        for (size_t core = 0; core < cores; core++)
        {
            Tensor* lossOverW = new Tensor();
            weightGradients[core].push_back(lossOverW);
            Tensor* lossOverb = new Tensor();
            biasGradients[core].push_back(lossOverb);
            Tensor* input = new Tensor(batchSize, height, width, inputChannels);
            inputs[core].push_back(input);
        }

        ReLULayer* layer = new ReLULayer();
        layers.push_back(layer);
    }

    void addAvgPoolLayer(size_t height, size_t width, size_t inputChannels, size_t stride)
    {
        Tensor* W = new Tensor();
        Tensor* GW = new Tensor();
        Tensor* MW = new Tensor();
        Tensor* VW = new Tensor();
        Tensor* b = new Tensor();
        Tensor* Gb = new Tensor();
        Tensor* Mb = new Tensor();
        Tensor* Vb = new Tensor();

        weights.push_back(W);
        Gweights.push_back(GW);
        Mweights.push_back(MW);
        Vweights.push_back(VW);
        biases.push_back(b);
        Gbiases.push_back(Gb);
        Mbiases.push_back(Mb);
        Vbiases.push_back(Vb);

        for (size_t core = 0; core < cores; core++)
        {
            Tensor* lossOverW = new Tensor();
            weightGradients[core].push_back(lossOverW);
            Tensor* lossOverb = new Tensor();
            biasGradients[core].push_back(lossOverb);
            Tensor* input = new Tensor(batchSize, height, width, inputChannels);
            inputs[core].push_back(input);
        }

        AvgPoolLayer* layer = new AvgPoolLayer(stride);
        layers.push_back(layer);
    }

    void addSoftLayer(size_t height, size_t width, size_t inputChannels)
    {
        Tensor* W = new Tensor();
        Tensor* GW = new Tensor();
        Tensor* MW = new Tensor();
        Tensor* VW = new Tensor();
        Tensor* b = new Tensor();
        Tensor* Gb = new Tensor();
        Tensor* Mb = new Tensor();
        Tensor* Vb = new Tensor();

        weights.push_back(W);
        Gweights.push_back(GW);
        Mweights.push_back(MW);
        Vweights.push_back(VW);
        biases.push_back(b);
        Gbiases.push_back(Gb);
        Mbiases.push_back(Mb);
        Vbiases.push_back(Vb);

        for (size_t core = 0; core < cores; core++)
        {
            Tensor* lossOverW = new Tensor();
            weightGradients[core].push_back(lossOverW);
            Tensor* lossOverb = new Tensor();
            biasGradients[core].push_back(lossOverb);
            Tensor* input = new Tensor(batchSize, height, width, inputChannels);
            inputs[core].push_back(input);
        }

        SoftLayer* layer = new SoftLayer();
        layers.push_back(layer);
    }

    void setLossFunction(size_t channels)
    {
        lossFunc = new CrossError();
        Tensor* W = new Tensor();
        Tensor* GW = new Tensor();
        Tensor* MW = new Tensor();
        Tensor* VW = new Tensor();
        Tensor* b = new Tensor();
        Tensor* Gb = new Tensor();
        Tensor* Mb = new Tensor();
        Tensor* Vb = new Tensor();

        weights.push_back(W);
        Gweights.push_back(GW);
        Mweights.push_back(MW);
        Vweights.push_back(VW);
        biases.push_back(b);
        Gbiases.push_back(Gb);
        Mbiases.push_back(Mb);
        Vbiases.push_back(Vb);

        for (size_t core = 0; core < cores; core++)
        {
            Tensor* lossOverW = new Tensor();
            weightGradients[core].push_back(lossOverW);
            Tensor* lossOverb = new Tensor();
            biasGradients[core].push_back(lossOverb);
            Tensor* input = new Tensor(batchSize, 1, 1, channels);
            inputs[core].push_back(input);
        }
    }

    void printGradients()
    {
        for (size_t ldx = 0; ldx < layers.size(); ldx++)
            weightGradients[0][ldx]->print();
    }
    
    std::vector<Tensor*> weights;
    std::vector<Tensor*> Gweights;
    std::vector<Tensor*> Mweights;
    std::vector<Tensor*> Vweights;
    std::vector<Tensor*> biases;
    std::vector<Tensor*> Gbiases;
    std::vector<Tensor*> Mbiases;
    std::vector<Tensor*> Vbiases;
    std::vector<std::vector<Tensor*>> weightGradients;
    std::vector<std::vector<Tensor*>> inputs;
    std::vector<std::vector<Tensor*>> biasGradients;
    CrossError* lossFunc;
    std::vector<LayerBase*> layers;
    size_t cores;
    size_t batchSize;
    double beta1 = 0.9;
    double beta2 = 0.999;
};