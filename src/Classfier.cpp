#include "Classfier.h"

/************************************************************************/
/*  Implementation for class CNeuralNetwork   */
/************************************************************************/
CNeuralNetwork::CNeuralNetwork()
{
	
}
CNeuralNetwork::CNeuralNetwork(int numOfClass)
{
	m_nnEngine = new FANN::neural_net();
	m_iNumOfClass = numOfClass;
}

CNeuralNetwork::~CNeuralNetwork()
{
	delete m_nnEngine;
}

void CNeuralNetwork::setNeuralNetworkParm(const SNeuralNetworkParameter& _param)
{
	UINT32* layers = new UINT32(_param.uiNumOfLayers);
	m_nnEngine->create_standard_array(_param.uiNumOfLayers,layers);
	m_nnEngine->set_learning_rate(_param.fLearingRate);
	m_nnEngine->set_activation_steepness_hidden(1.0);
	m_nnEngine->set_activation_steepness_output(1.0);
	m_nnEngine->set_activation_function_hidden(FANN::SIGMOID_SYMMETRIC_STEPWISE);
	m_nnEngine->set_activation_function_output(FANN::SIGMOID_SYMMETRIC_STEPWISE);
	m_nnEngine->set_training_algorithm(FANN::TRAIN_QUICKPROP);

	m_uiNumOfMaxIter = _param.uiNumOfMaxIter;
	m_fDesiredError  = _param.fDesiredError;
	delete[] layers;
}

void CNeuralNetwork::train()
{
	m_nnEngine->init_weights(m_nnTrainData);
	m_nnEngine->train_on_data(m_nnTrainData,m_uiNumOfMaxIter,0,m_fDesiredError);
}

// int CNeuralNetwork::classify(FLOAT32* _data)
// {
// 	FLOAT32* output = new FLOAT32[m_iNumOfClass];
// 	output = m_nnEngine->run(_data);
// 	for (int i = 0; i < m_iNumOfClass; i ++)
// 	{
// 		if (1 == output[i])
// 		{
// 			return i+1;
// 		}
// 	}
// 	return 0;
// }

int CNeuralNetwork::classify( const std::string _str)
{

}


void CNeuralNetwork::setTrainData(arma::mat _trainMat, arma::uvec _trainLabel)
{
	DOUBLE* p = _trainMat.memptr();
	UINT32  uiNumOfCols = _trainMat.n_cols;
	UINT32  uiNumOfRows = _trainMat.n_rows;

	//allocate input pointer array for fann port
	FLOAT32 **input;
	input = new FLOAT32*[uiNumOfCols];
	for (int i = 0; i <  uiNumOfCols; i ++)
	{
		input[i] = new FLOAT32[uiNumOfRows];
		input[i] = (FLOAT32*)_trainMat.colptr(i);
	}

	/**allocate output pointer array
	*  the format is as following:
	*
	*/
	FLOAT32 **output;
	output = new FLOAT32*[uiNumOfCols];
	for (int k = 0; k < uiNumOfCols; k ++)
	{
		output[k] = new FLOAT32[m_iNumOfClass];
		memset(output[k],0,m_iNumOfClass);
		int ind = _trainLabel[uiNumOfCols];
		output[k][ind-1] = 1;
	}

	//set train data via invoking fann lib port
	m_nnTrainData.set_train_data(uiNumOfCols, 
								 uiNumOfRows,input,
								 m_iNumOfClass,output);

	//delete allocated memory
	for(int j=0;j < uiNumOfCols;j++)
	{
		delete []input[j];
	}
	delete []input;
}

void CNeuralNetwork::saveModelToFile(const std::string _str)
{
	m_nnEngine->save_to_fixed(_str);
}

void CNeuralNetwork::loadModelFromFile(const std::string _str)
{
	m_nnEngine->create_from_file(_str);
}