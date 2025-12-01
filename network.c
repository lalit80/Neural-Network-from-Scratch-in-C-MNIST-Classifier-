#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include "myHeader.h"

#define SAMPLE_SIZE (28*28 + 10)

typedef struct InputData {
    double   x[28*28];           // pixel data      28x28
    double   y[10];              // correct digit   1x10 
}   InputData;

typedef struct Network {
    int         noOfLayers;
    int*        sizeOfLayer;
    double**    preActivation;
    double**    postActivation;
    double**    biasMatrix;
    double***   weightMatrix;
    double**    gradient_B;
    double***   gradient_W;
}   Network;

Network* getNetworkObject(int noOfLayers, int* sizes);
InputData* getTrainingData(const char* filename, int* count);
void SGD(Network* net, InputData* trainingData, int noOfSamples, int epochs, int miniBatchSize, double eta);
void updateMiniBatch(Network* net, InputData* trainingData, int batchSize, double eta);
void feedForward(Network* net, double* input);
void backPropagation(Network* net, double* input, double* output);
void evaluate(Network* net, InputData* testData, int noOfSamples);
void freeNetwork(Network* network);
void shuffle(InputData* data, int noOfSamples);

int main()
{
    int layers = 2;
    int sizes[2] = {784, 10};
    int epochs = 300;
    int miniBatchSize = 10;
    double eta = 1.0;
    int noOfSamples = 0;
    int trainCount = 0, testCount = 0;
    Network* network = 0;

    InputData* trainingData = getTrainingData("./data/mnist_train_data.bin", &trainCount);
    InputData* testData = getTrainingData("./data/mnist_test_data.bin", &testCount);
    if (!trainingData || trainCount == 0 || !testData || testCount == 0) {
        printf("Failed to load data\n");
        return 1;
    }
    printf("Loaded %d training samples, %d test samples\n", trainCount, testCount);

    network = getNetworkObject(layers, sizes);

    SGD(network, trainingData, trainCount, epochs, miniBatchSize, eta);
    evaluate(network, testData, testCount);

    freeNetwork(network);
    return 0;
}

void SGD(Network* net, InputData* trainingData, int noOfSamples, int epochs, int miniBatchSize, double eta)
{
    for (int i = 0; i < epochs; ++i) {
        shuffle(trainingData, noOfSamples);

        for (int start = 0; start < noOfSamples; start += miniBatchSize) {
            int end = (start + miniBatchSize < noOfSamples) ? (start + miniBatchSize) : noOfSamples;
            int batchSize = end - start;
            updateMiniBatch(net, &trainingData[start], batchSize, eta);
        }
        printf("\t%d epoch done\n", (i + 1));
    }
}

void updateMiniBatch(Network* net, InputData* trainingData, int batchSize, double eta)
{
    int L = net->noOfLayers;

    for (int l = 0; l < L - 1; ++l) {
        for (int j = 0; j < net->sizeOfLayer[l + 1]; ++j) {
            net->gradient_B[l][j] = 0.0;
            for (int k = 0; k < net->sizeOfLayer[l]; ++k) {
                net->gradient_W[l][j][k] = 0.0;
            }
        }
    }

    for (int i = 0; i < batchSize; ++i) {
        backPropagation(net, trainingData[i].x, trainingData[i].y);
    }

    // update the weights/biases
    for (int l = 0; l < L - 1; ++l) {
        for (int j = 0; j < net->sizeOfLayer[l + 1]; ++j) {
            net->biasMatrix[l][j] -= (eta / batchSize) * net->gradient_B[l][j];
            for (int k = 0; k < net->sizeOfLayer[l]; ++k)
                net->weightMatrix[l][j][k] -= (eta / batchSize) * net->gradient_W[l][j][k];
        }
    }

}

void feedForward(Network* net, double* input)
{
    copyArray(net->postActivation[0], input, net->sizeOfLayer[0]);

    for (int layer = 1; layer < net->noOfLayers; ++layer) {
        // calculate preactivation for all neuron in layer layer
        for (int j = 0; j < net->sizeOfLayer[layer]; ++j) {
            // a(l) = W . a(l - 1) + b
            // z = wa
            double z = vector_multiply_sum(net->postActivation[layer - 1], net->weightMatrix[layer - 1][j], net->sizeOfLayer[layer - 1]);
            
            // z = wa + b
            z += net->biasMatrix[layer - 1][j];         // add bias

            net->preActivation[layer - 1][j] = z;
            net->postActivation[layer][j] = sigmoid(z);
        }
    }
}

void backPropagation(Network* net, double* input, double* output)
{
    // gradients will contain summation of gradients of a mini batch
    int L = net->noOfLayers;

    feedForward(net, input);

    // error for Layer (L - 1)
    // delta^L = (a^L - y) . sigma'(z^L)
    for (int i = 0; i < net->sizeOfLayer[L - 1]; ++i) {
        double z = net->preActivation[L - 2][i];
        double a = net->postActivation[L - 1][i];
        double delta = (a - output[i]) * (sigmoid_derivative(z));

        // dC/db = delta^L
        net->gradient_B[L - 2][i] += delta;

        // dC/dW = delta^L(a^(L-1))^T
        for (int j = 0; j < net->sizeOfLayer[L - 2]; ++j) {
            net->gradient_W[L - 2][i][j] += delta * net->postActivation[L - 2][j];
        }
    }

    // error for hidden layers
    for (int l = L - 2; l > 0; --l) {
        for (int i = 0; i < net->sizeOfLayer[l]; ++i) {
            double sum = 0;
            for (int j = 0; j < net->sizeOfLayer[l + 1]; ++j) {
                sum += net->weightMatrix[l][j][i] * net->gradient_B[l][j];
            }
            net->gradient_B[l - 1][i] += sum * sigmoid_derivative(net->preActivation[l - 1][i]);

            for (int k = 0; k < net->sizeOfLayer[l - 1]; ++k) {
                net->gradient_W[l - 1][i][k] += net->gradient_B[l - 1][i] * net->postActivation[l - 1][k];
            }
        }
    }
}

void evaluate(Network* net, InputData* testData, int noOfSamples)
{
    int correct = 0;

    for (int i = 0; i < noOfSamples; ++i) {
        feedForward(net, testData[i].x);

        int predicted = 0, actual = 0;
        double max = net->postActivation[net->noOfLayers - 1][0];
        for(int j = 1; j < 10; ++j) {
            if (net->postActivation[net->noOfLayers - 1][j] > max) {
                max = net->postActivation[net->noOfLayers - 1][j];
                predicted = j;
            }
            if (testData[i].y[j] == 1.0) { actual = j; }
        }
        if (predicted == actual) ++correct;
    }
    printf("Accuracy: %d / %d (%.2f%%)\n", correct, noOfSamples, 100.0 * correct / noOfSamples);
    //return correct;
}

Network* getNetworkObject(int noOfLayers, int* sizes)
{
    Network* network = malloc(sizeof(Network));

    network->noOfLayers = noOfLayers;

    // size
    network->sizeOfLayer = malloc(noOfLayers * sizeof(int));
    for (int i = 0; i < noOfLayers; ++i) {
        network->sizeOfLayer[i] = sizes[i];
    }

    // bias-preActivation
    network->biasMatrix = malloc((noOfLayers - 1) * sizeof(double*));
    network->gradient_B = malloc((noOfLayers - 1) * sizeof(double*));
    network->preActivation = malloc((noOfLayers - 1) * sizeof(double*));
    for (int i = 1; i < noOfLayers; ++i) {
        network->biasMatrix[i - 1] = malloc(sizeof(double) * network->sizeOfLayer[i]);
        network->gradient_B[i - 1] = malloc(sizeof(double) * network->sizeOfLayer[i]);
        network->preActivation[i - 1] = malloc(sizeof(double) * network->sizeOfLayer[i]);
        for (int j = 0; j < network->sizeOfLayer[i]; ++j) {
            network->biasMatrix[i - 1][j] = rand_uniform();
            network->gradient_B[i - 1][j] = 0;
        }
    }

    // postactivation
    network->postActivation = malloc(noOfLayers * sizeof(double*));
    for (int i = 0; i < noOfLayers; ++i) {
        network->postActivation[i] = malloc(sizeof(double) * network->sizeOfLayer[i]);
    }
    
    // weight
    network->weightMatrix = malloc((noOfLayers - 1) * sizeof(double**));
    network->gradient_W = malloc((noOfLayers - 1) * sizeof(double**));
    for (int i = 0; i < noOfLayers - 1; ++i) {
        int rows = network->sizeOfLayer[i + 1];
        int cols = network->sizeOfLayer[i];
        network->weightMatrix[i] = malloc(sizeof(double*) * rows);
        network->gradient_W[i] = malloc(sizeof(double*) * rows);
        for (int j = 0; j < rows; ++j) {
            network->weightMatrix[i][j] = malloc(sizeof(double) * cols);
            network->gradient_W[i][j] = malloc(sizeof(double) * cols);
            for (int k = 0; k < cols; ++k) {
                network->weightMatrix[i][j][k] = rand_uniform();
                network->gradient_W[i][j][k] = 0;
            }
        }
    }

    return network;
}

void freeNetwork(Network* network)
{
    for (int i = 0; i < network->noOfLayers - 1; ++i) {
        for (int j = 0; j < network->sizeOfLayer[i + 1]; ++j) {
            free(network->weightMatrix[i][j]);
            free(network->gradient_W[i][j]);
        }
        free(network->weightMatrix[i]);
        free(network->gradient_W[i]);
        free(network->gradient_B[i]);
        free(network->biasMatrix[i]);
        free(network->preActivation[i]);
    }

    for (int i = 0; i < network->noOfLayers; ++i) {
        free(network->postActivation[i]);
    }

    free(network->weightMatrix);
    free(network->biasMatrix);
    free(network->gradient_B);
    free(network->gradient_W);
    free(network->preActivation);
    free(network->postActivation);
    free(network->sizeOfLayer);
    free(network);
}

InputData* getTrainingData(const char* filename, int* count)
{
    FILE *file = NULL;
    size_t num_samples = 0;
    InputData *data = NULL;
    
    *count = 0;

    // Open the binary file in read-binary mode
    file = fopen(filename, "rb");
    if (file == NULL) {
        fprintf(stderr, "Error opening file %s: %s\n", filename, strerror(errno));
        return NULL;
    }

    // 1. Read the total number of samples (The first 4 bytes are the integer count)
    if (fread(&num_samples, sizeof(int), 1, file) != 1) {
        fprintf(stderr, "Error reading sample count from %s. File might be empty or corrupted.\n", filename);
        fclose(file);
        return NULL;
    }

    // Check for a non-negative, sensible count
    if (num_samples <= 0) {
        fprintf(stderr, "File %s contains zero or negative sample count (%d).\n", filename, num_samples);
        fclose(file);
        return NULL;
    }

    // 2. Allocate memory for all samples
    // InputData struct has the exact memory layout (784+10 doubles) as the binary data
    size_t required_bytes = num_samples * sizeof(InputData);
    data = (InputData*)malloc(required_bytes);
    if (data == NULL) {
        perror("Failed to allocate memory for training data");
        fclose(file);
        return NULL;
    }

    // 3. Read the entire data block at once for efficiency
    size_t elements_read = fread(data, sizeof(InputData), num_samples, file);

    if (elements_read != num_samples) {
        fprintf(stderr, "Warning: Expected %d samples, but only read %zu from the file.\n", num_samples, elements_read);
    }

    // Success
    *count = (int)elements_read;
    fclose(file);
    return data;
}

void shuffle(InputData* data, int noOfSamples)
{
    for (int i = noOfSamples - 1; i > 0; --i) {
        int j = rand() % (i + 1);
        InputData tmp = data[i];
        data[i] = data[j];
        data[j] = tmp;
    }
}

/*
    50000 training samples, 10000 test samples

    int layers = 3;
    int sizes[3] = {784, 16, 10};
    int epochs = 300;
    int miniBatchSize = 10;
    double eta = 3.0;
    Accuracy: 8950 / 10000 (89.50%)                 (30 - 35 minutes to train)

    int layers = 3;
    int sizes[3] = {784, 64, 10};
    int epochs = 200;
    int miniBatchSize = 10;
    double eta = 1.0;
    Accuracy: 9465 / 10000 (94.65%)                 (110 minutes)

    int layers = 3;
    int sizes[3] = {784, 16, 10};
    int epochs = 300;
    int miniBatchSize = 10;
    double eta = 0.5;
    Accuracy: 9237 / 10000 (92.37%)                 (~45 minutes)

    int layers = 3;
    int sizes[3] = {784, 24, 10};
    int epochs = 400;
    int miniBatchSize = 10;
    double eta = 0.1;
    Accuracy: 9424 / 10000 (94.24%)                 (100 minutes)

    int layers = 2;
    int sizes[2] = {784, 10};
    int epochs = 200;
    int miniBatchSize = 10;
    double eta = 1.0;
    Accuracy: 9214 / 10000 (92.14%)                 (15 minutes)

    int layers = 4;
    int sizes[4] = {784, 128, 64, 10};
    int epochs = 35;
    int miniBatchSize = 10;
    double eta = 0.05;
    Accuracy: 9312 / 10000 (93.12%)
*/
