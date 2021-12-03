//////////////////////////////////////////////////////////////////////////
/* COPYRIGHT NOTICE
 * Copyright (c) 
 * All rights reserved.
*/ 
//////////////////////////////////////////////////////////////////////////

/// @file 
/// @brief Test program for test file reader
/// 
/// 
/// 
/// @version 1.0
/// @author bob
/// @date 12/01/2021

#include <string>
#include <map>
#include <vector>
#include <cstring>
#include "DataReader.h"
#include "DataProc.h"
#include "FeatureExtractor.h"


int main()
{
    DataProc::parameters param;
	param.uiSlideWinWidth = 512;
	param.uiSlideWinStep  = 256;
    DataProc    dataproc(param);
    CDataReader datareader;
    printf("--------------------");

//    std::string basePath = "Datasets/train/";
    std::string basePath = "/home/westwell/Documents/project/activity/demos/Datasets/train/";

	if (datareader.readDir(basePath))
	{
		std::vector<std::string> dirlist  = datareader.getDirsList();
		for (int i = 0; i < dirlist.size();i ++)
		{
			//std::string str(dirlist[i]);
//			dirlist[i] = basePath + dirlist[i];
			
			if (datareader.readDir(dirlist[i]))
			{
				std::vector<std::string> filelist = datareader.getFilesList();

				if (!filelist.empty())
				{
					for (int j = 0; j < filelist.size(); j ++)
					{
//						filelist[j] = dirlist[i] + "/" + filelist[j];
						if (datareader.readFile(filelist[j]))
						{
							arma::mat fileMat = datareader.getFileData();
							dataproc.setDataMatirx(fileMat);
							if (dataproc.segment())
							{
								arma::cube stream = dataproc.getSegmentedAcceleraterStream();
								for (UINT32 i = 0; i < stream.n_slices; i ++)
								{
									FeatureExtractor featureEngine;
									featureEngine.setRawStream(stream.slice(i));
									// featureEngine.extract(0);
									// featureEngine.extract(1);
									// featureEngine.extract(2);
									// featureEngine.extract(3);
									// featureEngine.extract(4);
									featureEngine.extract(5);
									featureEngine.extract(7);
									//featureEngine.getExtractedFeatures().print("Extracted Feature:");
									dataproc.concatExtractedFeatureToMat(featureEngine.getExtractedFeatures());
								}
							}
							else
							{
								continue;
							}
						}
					}  //end for 
				} //end readdir
			}  // end for 
		}
		return 1;

	} 
	else
	{
		return 0;
	}
}