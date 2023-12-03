#include <iostream>
#include "Tensor.h"
#include "Model.h"
#include <string>
#include "Plot.h"
Tensor readImages(std::string path);
std::vector<unsigned char> readLabels(std::string path);
Tensor filterLabels(std::vector<unsigned char> labels);

int main()
{
    Tensor images = readImages("train-images.idx3-ubyte");
    auto labels = readLabels("train-labels.idx1-ubyte");
    Tensor outputs = filterLabels(labels);

    Tensor testImages = readImages("t10k-images.idx3-ubyte");
    auto testLabels = readLabels("t10k-labels.idx1-ubyte");
    Tensor testOutputs = filterLabels(testLabels);

    Model lolModel(8, 32);
    lolModel.addConvLayer(3, 1, 32, 28, 28);
    lolModel.addAvgPoolLayer(26, 26, 32, 2);
    lolModel.addReLULayer(13, 13, 32);
    lolModel.addConvLayer(4, 32, 64, 13, 13);
    lolModel.addAvgPoolLayer(10, 10, 64, 2);
    lolModel.addReLULayer(5, 5, 64);
    lolModel.addConvLayer(5, 64, 10, 5, 5);
    lolModel.addSoftLayer(1, 1, 10);

    lolModel.setLossFunction(10);

    //lolModel.loadModel("models5\\weights30");
    
    size_t timestep = 0;
    for (size_t epoch = 0; epoch < 50; epoch++)
    {
        std::cout << "Epoch " << epoch << std::endl;
        std::cout << "Loss: " << lolModel.totalLossMulti(testImages, testOutputs) << std::endl;
        timestep = lolModel.trainMulti(images, outputs, 0.001, timestep);
        std::string path = "models8\\weights" + std::to_string(epoch);
        lolModel.saveModel(path);
    }

    std::cout << std::endl;
    auto predictions = lolModel.predict(testImages, 0);
    size_t correct = 0;
    for (size_t bdx = 0; bdx < testImages.gB(); bdx++)
    {
        size_t numberIndex = predictions.maxChannelIndex(bdx, 0, 0);
        std::string path = "images\\" + std::to_string(numberIndex) + "_" + std::to_string(testLabels[bdx]) + "_" + std::to_string(bdx) + ".ppm";
        if (numberIndex == testLabels[bdx])
            correct++;
    }
    std::cout << "Correct ratio: " << double(correct) / double(testImages.gB()) << std::endl;


}

int32_t ntohl(int32_t val)
{
    auto data = reinterpret_cast<char*>(&val);
    std::reverse(data, data + sizeof(val));
    return val;
}

Tensor readImages(std::string path)
{
    FILE* fptr;
    fopen_s(&fptr, path.c_str(), "rb");
    int32_t magicLabelNumber = 0;
    int32_t numberOfImages = 0;
    int32_t rows = 0;
    int32_t cols = 0;
    size_t readBytes = fread(&magicLabelNumber, sizeof(int32_t), 1, fptr);
    readBytes = fread(&numberOfImages, sizeof(int32_t), 1, fptr);
    readBytes = fread(&rows, sizeof(int32_t), 1, fptr);
    readBytes = fread(&cols, sizeof(int32_t), 1, fptr);

    magicLabelNumber = ntohl(magicLabelNumber);
    numberOfImages = ntohl(numberOfImages);

    Tensor result(numberOfImages, 28, 28, 1);

    rows = ntohl(rows);
    cols = ntohl(cols);

    std::cout << rows << std::endl;
    std::cout << cols << std::endl;

    uint8_t data{};
    for (int32_t imageIndex = 0; imageIndex < numberOfImages; imageIndex++)
    {
        size_t index = 0;
        for (int32_t rowIndex = 0; rowIndex < rows; rowIndex++)
        {
            for (int32_t colIndex = 0; colIndex < rows; colIndex++)
            {
                readBytes = fread(&data, sizeof(uint8_t), 1, fptr);
                double datadouble = static_cast<double>(data) / 255.0;
                result(imageIndex, rowIndex, colIndex, 0) = datadouble;
            }
        }
    }
    fclose(fptr);
    return result;
}

std::vector<unsigned char> readLabels(std::string path)
{
    std::vector<unsigned char> labels;

    FILE* fptr;
    fopen_s(&fptr, path.c_str(), "rb");
    int32_t magicLabelNumber = 0;
    int32_t numberOfLabels = 0;
    size_t readBytes = fread(&magicLabelNumber, sizeof(int32_t), 1, fptr);
    readBytes = fread(&numberOfLabels, sizeof(int32_t), 1, fptr);

    magicLabelNumber = ntohl(magicLabelNumber);
    numberOfLabels = ntohl(numberOfLabels);

    uint8_t data{};
    for (int32_t labelIndex = 0; labelIndex < numberOfLabels; labelIndex++)
    {
        readBytes = fread(&data, sizeof(uint8_t), 1, fptr);
        labels.push_back(data);
    }
    fclose(fptr);

    return labels;
}

Tensor filterLabels(std::vector<unsigned char> labels)
{
    Tensor result(labels.size(), 1, 1, 10);
    size_t index = 0;
    double part = 1.0 / 100.0;
    for (auto label : labels)
    {
        
        for (size_t cdx = 0; cdx < 10; cdx++)
        {
            if (static_cast<size_t>(label) == cdx)
                result(index, 0, 0, cdx) = 1.0 - part;
            else
                result(index, 0, 0, cdx) = part / 9.0;
        }
        index++;
    }
    return result;
}