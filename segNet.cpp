#include "NvCaffeParser.h"
#include "NvInfer.h"
#include "stdio.h"
#include "cudaMappedMemory.h"
#include "loadImage.h"
#include <sys/time.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>
#include <memory>
#include <stdlib.h>
#include <NvInfer.h>
#include <cstddef>
#include <cstdint>
#include <assert.h>

uint64_t current_timestamp(){
	struct timeval te;
	gettimeofday(&te, NULL);
	return te.tv_sec*1000LL+te.tv_usec/1000;
}


int main()
{	
	class Logger:public nvinfer1::ILogger
        {
		void log(Severity severity, const char* msg)override
		{
			if(severity!=Severity::kINFO)	
				std::cout<<msg<<std::endl;
		}
	}gLogger;
	/*
	nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(gLogger);
	nvinfer1::INetworkDefinition* network= builder->createNetwork();
	nvcaffeparser1::ICaffeParser* parser = nvcaffeparser1::createCaffeParser();
	const nvcaffeparser1::IBlobNameToTensor* blobNameToTensor = parser->parse("../models/city4/1_CityNet_4_remove_power.prototxt",
			"../models/city4/CityNet_4_remove_power.caffemodel", *network, nvinfer1::DataType::kFLOAT);
	nvinfer1::ITensor* test = blobNameToTensor->find("conv_score");
	//printf(test->getDimensions().nbDims);
	//int a = 1;
	network->markOutput(*blobNameToTensor->find("conv_score"));
	std::cout<<test->getDimensions().d[0]<<","<<test->getDimensions().d[1]<<","<<test->getDimensions().d[2]<<std::endl;
	
	//add mean subtraction
	nvinfer1::Dims inputDims = network->getInput(0)->getDimensions();
	std::cout<<inputDims.nbDims<<","<<inputDims.d[0]<<","<<inputDims.d[1]<<","<<inputDims.d[2]<<std::endl;
	std::cout<<inputDims.d[0]*inputDims.d[1]*inputDims.d[2]<<std::endl;
	*/
	/*
	int meanArray[inputDims.d[0]*inputDims.d[1]*inputDims.d[2]];
	std::cout<<"debug0"<<std::endl;
	for(int i=0;i<inputDims.d[1]*inputDims.d[2];i++){
		std::cout<<i<<std::endl;
		meanArray[i] = 104.00;
		if(i==inputDims.d[1]*inputDims.d[2]-1)
			std::cout<<"debug0"<<std::endl;
	}
	for(int i=0;i<2*inputDims.d[1]*inputDims.d[2];i++){
                meanArray[i] = 117.00;
        }
	for(int i=0;i<3*inputDims.d[1]*inputDims.d[2];i++){
                meanArray[i] = 123.00;
        }
	std::cout<<"debug1"<<std::endl;
	nvinfer1::Weights meanWeights{nvinfer1::DataType::kFLOAT,meanArray,inputDims.d[0]*inputDims.d[1]*inputDims.d[2]};
	auto mean = network->addConstant(nvinfer1::Dims3(inputDims.d[0], inputDims.d[1], inputDims.d[2]), meanWeights);
    	auto meanSub = network->addElementWise(*network->getInput(0), *mean->getOutput(0), nvinfer1::ElementWiseOperation::kSUB);
	std::cout<<"debug2"<<std::endl;
	std::cout<<network->getLayer(0)->getNbInputs()<<std::endl;
	network->getLayer(0)->setInput(0, *meanSub->getOutput(0));
	*/
	//std::ifstream cache("../models/segNet.engine");	
	/*
	//create engine
	builder->setMaxBatchSize(2);
	builder->buildCudaEngine(*network);
	builder->setMaxWorkspaceSize(1 << 20);
	std::cout<<"create engine, wait..."<<std::endl;
	nvinfer1::ICudaEngine* engine = builder->buildCudaEngine(*network);
	std::cout<<"create complete!"<<std::endl;
	assert(network->getNbInputs() == 1);
	nvinfer1::Dims mInputDims = network->getInput(0)->getDimensions();
    	assert(mInputDims.nbDims == 3);

	//serialize
	std::stringstream modelStream;
	modelStream.seekg(0,modelStream.beg);
	std::cout<<"serialize engine"<<std::endl; 
	nvinfer1::IHostMemory* serializedModel = engine->serialize();
	modelStream.write((const char*)serializedModel->data(), serializedModel->size());
	std::cout<<"size: "<<serializedModel->size()<<std::endl;
	std::ofstream outfile;
	outfile.open("../models/segNet.engine");
	std::cout<<"write serializedModel to disk"<<std::endl;
	outfile<<modelStream.rdbuf();
	std::cout<<"write complete!"<<std::endl;
	outfile.close();
	serializedModel->destroy();
	//invinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
	*/
	//deserialized 
	std::ifstream cache("../models/segNet.engine");
	std::stringstream modelStream;
	modelStream.seekg(0,modelStream.beg);
	modelStream<<cache.rdbuf();
	cache.close();
	nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
	modelStream.seekg(0, std::ios::end);
	const int modelSize = modelStream.tellg();
	modelStream.seekg(0, std::ios::beg);
	void* modelMem = std::malloc(modelSize);
	modelStream.read((char*)modelMem, modelSize);
	nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(modelMem, modelSize, NULL);
	free(modelMem);	
	//perform inference
	nvinfer1::IExecutionContext *context = engine->createExecutionContext();
	int inputIndex = engine->getBindingIndex("data");
	int outputIndex = engine->getBindingIndex("conv_score");
	std::cout<<inputIndex<<","<<outputIndex<<std::endl;
	nvinfer1::Dims inputDims = engine->getBindingDimensions(inputIndex);
	nvinfer1::Dims outputDims = engine->getBindingDimensions(outputIndex);
	size_t inputSize = 2*inputDims.d[0]*inputDims.d[1]*inputDims.d[2]*sizeof(float);
	size_t outputSize = 2*outputDims.d[0]*outputDims.d[1]*outputDims.d[2]*sizeof(float);
	void* inputCPU = NULL;
	void* inputCUDA = NULL;
	void* outputCPU = NULL;
	void* outputCUDA = NULL;
	cudaAllocMapped((void**)&inputCPU, (void**)&inputCUDA, inputSize);
	cudaAllocMapped((void**)&outputCPU, (void**)&outputCUDA, outputSize);
	cudaError_t cudaPreImageNet(float4* input, size_t inputWidth, size_t inputHeight, float* output, size_t outputWidth, size_t outputHeight, cudaStream_t stream);
	float* imgCPU=NULL;
	float* imgCUDA=NULL;
	int imgWidth=0;
	int imgHeight=0;
	loadImageRGBA("../data/test_image.jpg", (float4**)&imgCPU, (float4**)imgCUDA, &imgWidth, &imgHeight);
	printf("(%zu)\n",current_timestamp());
	cudaStream_t stream=NULL;
	cudaPreImageNet((float4*)imgCUDA, imgWidth, imgHeight, (float*)inputCUDA, inputDims.d[1], inputDims.d[2], stream);
	void* buffers[]={inputCUDA, outputCUDA};
	context->execute(1,buffers);
	printf("finish (%zu)\n", current_timestamp());
	printf("hello world!\n");
//	printf(nvinfer1::ILogger::Severity);
}
