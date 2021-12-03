//////////////////////////////////////////////////////////////////////////
/* COPYRIGHT NOTICE
 * Copyright (c) 
 * All rights reserved.
*/ 
//////////////////////////////////////////////////////////////////////////

/// @file 
/// @brief The SVM classifier class  
/// 
/// The svm classifier which wraps accord.net framework .
/// 
/// @version 1.0
/// @author bob
/// @date 10/16/2021

#pragma once
#include "globaltype.h"
#include "DataProc.h"
#include "floatfann.h"
#include "fann_cpp.h"

/** Struct for neural network parameters
	 *
	 *\note The struct is defined here because it can't be define in a managed class.
	 */
	struct SNeuralNetworkParameter
	{
		
		UINT32      uiNumOfLayers;   /**< number of layers */
		UINT32      uiNumOfInput;    /**< the number of neurons in input layer   */
		UINT32      uiNumOfHidden;      /**< the number of neurons in hidden layer   */
		UINT32      uiMumOfOutput;     /**< the number of neurons in output layer */
		FLOAT32     fDesiredError;     /**< the desired error crition */
		UINT32      uiNumOfMaxIter;    /**< the maximum iteration number */
		FLOAT32     fLearingRate;      /**< learning rate */
		 
		SNeuralNetworkParameter()
		{
			
		}
	};

/** The neural network class 
 *
 *	The class is used to implement neural network framework.
 *	\note This class is the wrapper of FANN lib. 
          For more information please refer to <a href="http://leenissen.dk/fann/wp/">Fast Artificial Neural Network Library</a> online.
 */

class CNeuralNetwork 
{
public:

	/** Class constructor
	 *
	 */
	CNeuralNetwork(int numOfClass);
	CNeuralNetwork();
	~CNeuralNetwork();

	/** Trainning function for neural network
	 *
	 *  train neural network via some predefined param and train data 
	 **/
	virtual void train();

    /** Test function for neural network
	 *
	 *  test neural network via test data 
	 **/

	// virtual int classify(FLOAT32* _data);

     /** Test function for neural network
	 *
	 *  test neural network via a file 
	 **/

	int classify( const std::string _str);

	/** Set neural network parameters
	 *
	 *  @param param the parameters needed by nn
	 *  \note refer to struct SNeuralNetworkParameter for parameters list
	 *  \sa SNeuralNetworkParameter
	 */ 
	void setNeuralNetworkParm(const SNeuralNetworkParameter& param);

	/** Set data source for train 
	 *
	 * The data is obtained from HDF5 file. 
	 * Furthermore, the data is further transformed into reduced dimension via DR method such as PCA\KPCA\KDA.
	 * The data matrix represent train data set. The number of row denotes feature dimension and the column number of matrix is the number of train sample.
	 * The sample label is a column vector in which each entry is the label
	 * @param _trainMat reduced train data 
	 * @param _trainLabel the corresponding sample label 
	 */
	void setTrainData(arma::mat _trainMat, arma::uvec _trainLabel);

	void saveModelToFile(const std::string _str);
	void loadModelFromFile(const std::string _str);


private:

	FANN::neural_net *m_nnEngine;
	FANN::training_data m_nnTrainData;

private:

    SNeuralNetworkParameter param_;
	UINT32      m_uiNumOfLayers;   /**< number of layers */
	UINT32      m_iNumOfClass;     /**< number of class */
	UINT32      m_uiNumOfInput;    /**< the number of neurons in input layer   */
	UINT32      m_uiNumOfHidden;      /**< the number of neurons in hidden layer   */
	UINT32      m_uiMumOfOutput;     /**< the number of neurons in output layer */
	FLOAT32     m_fDesiredError;     /**< the desired error crition */
	UINT32      m_uiNumOfMaxIter;    /**< the maximum iteration number */
	FLOAT32     m_fLearingRate;      /**< learning rate */

};