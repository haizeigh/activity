//////////////////////////////////////////////////////////////////////////
/* COPYRIGHT NOTICE
 * Copyright (c) 
 * All rights reserved.
*/ 
//////////////////////////////////////////////////////////////////////////

/// @file 
/// @brief Test program for test file reader
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
#include "include/ActivityRecognition/DataReader.h"


int main1(int argc, char** argv)
{
//  FLAGS_logtostderr=1;
//  google::InitGoogleLogging(argv[0]);
  CDataReader datareader;
  std::string baseDir = "/home/westwell/CLionProjects/activity1/demos/Datasets/train";
  datareader.readDir(baseDir);
  std::vector<std::string> dirLists = datareader.getDirsList();
  std::vector<std::string> filelists;
//  LOG(INFO)<< dirLists.size();

  for (int i=0; i < dirLists.size(); i ++)
  {
    datareader.readDir(dirLists[i]);
    filelists = datareader.getFilesList();
    
    if (!filelists.empty())
		{
					for (int j = 0; j < filelists.size(); j ++)
					{
//            LOG(INFO)<< filelists[j];
            if (datareader.readFile(filelists[j]))
						{
              arma::mat fileMat = datareader.getFileData();
              std::cout<< "" <<std::endl;
//              LOG(INFO) << fileMat.n_rows << "*" << fileMat.n_cols;
              //fileMat.print();
            }
          }
    }
  }
	
}