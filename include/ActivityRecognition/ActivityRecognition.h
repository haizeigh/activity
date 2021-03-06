//////////////////////////////////////////////////////////////////////////
/* COPYRIGHT NOTICE
 * Copyright (c) 
 * All rights reserved.
*/ 
//////////////////////////////////////////////////////////////////////////

/// @file 
/// @brief The main recognition class 
/// 
/// The main algorithm flowchart class which contains every step for implementing recognition such as train,classify etc.
/// 
/// @version 1.0
/// @author bob
/// @date 10/16/2021

#include <string>
#include <map>
#include <vector>
#include <cstring>

// #include "GRT.h"
//#include "globaltype.h"
 /** This class is main class for recognition which gets involved in each step
  * 
  * The main export port for outside invocation. The following example illustrates how to use the class.
  * \code{.cpp}
  * CActivityRecognition::_sParameters param;
  * param.sBasePath       = "..\\Datasets\\train";
  * param.uiNumOfType     = 2;
  *	param.sClassifierType = "SVM";
  *	param.sDrType         = "KDA";
  *	CActivityRecognition ar();
  * ar.setParam(param);
  * ar.loadTrainData("");
  * ar.train();
  * ar.classify();
  *	int *p  = ar.classify("test.h5",length);
  * \endcode
  */

class CActivityRecognition
{
public:

	struct SNNParameters 
	{
		  unsigned int num_input;
	      unsigned int num_output;
	      unsigned int num_layers;
	      unsigned int num_neurons_hidden;
	      float desired_error;
	      unsigned int max_epochs;
	      unsigned int epochs_between_reports;
	};

	/** Input parameters struct for the whole algorithm 
	 *
	 *  The parameter struct is used to provide an interface for adjusting algorithm implementation
	 */
	 struct SParameters
	{
		std::string sBasePath;     /**< base path for training data */
		std::string sDrType;       /**< dimension reduction type (PCA,KPAC,KDA) */
		unsigned int      uiNumOfType;   /**< number of posture types */
		unsigned int      uiWinWidth;    /**< width of slide window   */
		unsigned int      uiWinStep;      /**< step of slide window   */
		unsigned int      uiMaxOrder;     /**< maximum order of autoregressive model */
		unsigned int      uiDataLength;   /**< dimension of data  */
		SNNParameters SNNparam;
		SParameters()
		{
			uiNumOfType = 4;
			uiWinWidth  = 512;
			uiWinStep   = 256;
			uiMaxOrder  = 10;
		}
	};

	struct SAccs 
	{
		float x;
		float y;
		float z;
	};
	typedef std::list<SAccs> sAccData; 

	/**  constructor of class
   */
	CActivityRecognition(SParameters _param);

	~CActivityRecognition();

	void setTrainData();



	 /**
       * Use this function to set the default base path of data
       */
     inline void setDefaultBasePath(const std::string s) { m_basepath = s; }

	/**
     * @return the default base path
     */
     inline std::string getBasePath() { return m_basepath; }

	 /**
	 * The inline function for getting transformed features matrix like M*(C-1)
	 */
	 //inline arma::mat getTransformedFeatures(){return m_mTransformedFeatures;}

	 /**
	 * The inline function for getting raw class labels
	 */
	// inline arma::uvec getLabels(){return m_vTrainLabels;}

	/**
	 * train the classifier through reading file
	 * @param str the created train data file name 
	 */
        void train();

	/**
	* classify the activity from test file
	*/
	int* classify(const std::string& _str,int& num);

	/**
	 * classify the activity from real time accelerates database 
	 * @param _dataList list structure from db
	 * @return the classification string
	 */

	void classify(const std::vector<float>& _dataList);

	/**
	*  load training data from classification file format 
	*  @param trainningFile The file 
	*/
	void loadTrainData(const std::string& _file);

	/**
	*  Create train data from raw data in a director 
	*  save the data into fann input data format
	*  @param _filestr
	*/
	static bool createTrainData(const std::string& _filestr);

	/**
	    The function is used to create UCI dataset 
	 *   @param _file the trained file
	 *   @param _index  the index expected to be trained 
	 *   @note The override member is specifically for UCI HAR dataset
	 **/
	unsigned int createTrainData(const std::string& _file,const std::vector<int> _index);

	 std::string getClassifierType(){return m_sClassifierType;}

	 void saveModelToFile(const std::string _str,const std::string _type);
	 void loadModelFromFile(const std::string _str,const std::string _type);
	 
	/**
	 * Sets the classifier at the core of the pipeline.  A pipeline can only have one classifier or regressifier, 
	    setting a new classifier will override any previous classifier or regressifier.
          @param const Classifier& _classifier: a reference to the classifier module you want to add to the pipeline
           @return returns true if the classifier module was set successfully, false otherwise
	 */
	//  bool setClassifier(const Classifier& _classifier);
	
private:
	void setParam(const SParameters& _param);

private:

   	 std::string m_basepath;   /// base path for train data
	int m_sampleRate;                /// sample rate of accelerator data
	int m_offset;                    /// offset of slide window
	unsigned int m_uiNumOfType;
	unsigned int m_uiWinWidth;
	unsigned int m_uiWinStep;

	std::string m_sDrType;
	std::string m_sClassifierType; 

	unsigned int m_uiDataLength;

	SNNParameters m_sNNParam;
	

};
