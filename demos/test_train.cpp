//////////////////////////////////////////////////////////////////////////
/* COPYRIGHT NOTICE
 * Copyright (c) 
 * All rights reserved.
*/ 
//////////////////////////////////////////////////////////////////////////

/// @file 
/// @brief Test program for test datadoc
/// @version 1.0
/// @author bob
/// @date 12/01/2021

#include <string>
#include <map>
#include <vector>
#include <cstring>

#include <dirent.h>
//#include <glog/logging.h>
//#include "glog/raw_logging.h"
#include "DataReader.h"
#include "DataProc.h"
#include "FeatureExtractor.h"
#include "FeatureAnalyse.h"
//#include "fann.h"

//#include "doublefann.h"
//#include "fann_cpp.h"
#include "Classfier.h"


int main(int argc, char** argv)
{
//    printf("test_train");
//  FLAGS_logtostderr = 1;
//  FLAGS_log_dir = "./data";
//  google::InitGoogleLogging(argv[0]);
  CDataReader datareader;
  std::string baseDir = "/home/westwell/Documents/project/activity/demos/Datasets/train";
  datareader.readDir(baseDir);
  std::vector<std::string> dirLists = datareader.getDirsList();
  std::vector<std::string> filelists;

  DataProc::parameters param;
	param.uiSlideWinWidth = 512;
	param.uiSlideWinStep  = 256;
  DataProc  dataproc(param);

  PCAFeatureAnalyse*  pcaEngine;

  for (int i=0; i < dirLists.size(); i ++)
  {
    datareader.readDir(dirLists[i]);
    filelists = datareader.getFilesList();
    
    if (!filelists.empty())
		{
					for (int j = 0; j < filelists.size(); j ++)
					{
            if (datareader.readFile(filelists[j]))
						{
              arma::mat fileMat = datareader.getFileData();
                            dataproc.filterMV(fileMat);
                            fileMat.save("dataproc.txt");

              dataproc.setDataMatirx(fileMat);
							if (dataproc.segment())
							{
								arma::cube stream = dataproc.getSegmentedAcceleraterStream();
								for (UINT32 i = 0; i < stream.n_slices; i ++)
								{
									FeatureExtractor featureEngine;
									featureEngine.setRawStream(stream.slice(i));
									featureEngine.extract(0);
								  featureEngine.extract(1);
//								  featureEngine.extract(2);
								  featureEngine.extract(3);
								//	featureEngine.extract(4);
									//featureEngine.extract(5);
									//featureEngine.extract(7);
								//	featureEngine.getExtractedFeatures().print("Extracted Feature:");
									dataproc.concatExtractedFeatureToMat(featureEngine.getExtractedFeatures());
								}
							}
							else
							{
								continue;
							}
            }
          }
    }
    arma::mat featMat = dataproc.getExtractedFeatures();
    featMat.save("1.csv",csv_ascii);

    pcaEngine = new PCAFeatureAnalyse(featMat,6);
    pcaEngine->apply();

      featMat.save("train.data");

    //arma::mat tranMat = pcaEngine->getTransformationFeats();
    arma::mat tranMat = featMat;
//    arma::vec outVec(tranMat.n_cols, fill::zeros);
    arma::mat outVec(tranMat.n_rows, tranMat.n_cols, fill::zeros);


    FANN::training_data data;
    double* temp = tranMat.memptr();
    fann_type* input = (fann_type*)tranMat.memptr();
    fann_type* out = (fann_type*)outVec.memptr();

    //todo error
//      printf(tranMat.n_rows);
//      printf(tranMat.n_cols);
      std::cout<<tranMat.n_cols<<std::endl;
      std::cout<<tranMat.n_rows<<std::endl;
    data.set_train_data(tranMat.n_cols,tranMat.n_rows, &input,1 , &out);
    std::string str="train.data";
    data.save_train(str);

//    LOG(INFO) << featMat.n_rows << " * " << featMat.n_cols;
//    LOG(INFO) << tranMat.n_rows << " * " << tranMat.n_cols;

      SNeuralNetworkParameter param;

      param.uiNumOfLayers = 3;   /**< number of layers */
      param.uiNumOfInput = 32;    /**< the number of neurons in input layer   */
      param.uiNumOfHidden = 96;      /**< the number of neurons in hidden layer   */
      param.uiMumOfOutput =2;     /**< the number of neurons in output layer */
      param.fDesiredError = 0.001f;     /**< the desired error crition */
      param.uiNumOfMaxIter = 50000;    /**< the maximum iteration number */
      param.fLearingRate = 0.001;      /**< learning rate */


      CNeuralNetwork network;
      network.setNeuralNetworkParm(param);
      network.train();


      network.classify("");



  }
	
}