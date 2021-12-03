#include "DataReader.h"

CDataReader::CDataReader()
{
	m_vFileNames.clear();
	m_vDirNames.clear();
}

BOOL8 CDataReader::readDir(std::string dirName)
{
	DIR *dir;
	struct dirent *ent;
	struct stat st;    
    char p[1024] = {0};
	//std::string baseDir = "/home/westwell/source_codes/Mole/Mole/modules/ActivityRecognition/demos/Datasets/train";

	if ((dir = opendir(dirName.c_str())) != NULL) {
  	/* print all the files and directories within directory */
  		while ((ent = readdir (dir)) != NULL) {
        if((!strncmp(ent->d_name, ".", 1)) || (!strncmp(ent->d_name, "..", 2)))
            continue;

		if(ent->d_type & DT_DIR)
		{
			std::string tempDir = dirName + "/" + std::string(ent->d_name);
			m_vDirNames.push_back(tempDir);
		}
		else
		{
			std::string tempFile = dirName + "/" + std::string(ent->d_name);
			m_vFileNames.push_back(tempFile);
		}      
    }
    closedir (dir);
	} else {
 	 /* could not open directory */
  	perror ("");
  	return EXIT_FAILURE;
	}
	return true;
}

BOOL8 CDataReader::readFile(const std::string file)
{
	if (!m_vFileNames.empty())
	{
		m_vFileNames.clear();
	}
	if (!m_mFileData.is_empty())
	{
		m_mFileData.reset();
	}
	if (mlpack::data::Load(file,m_mFileData,false))
	{
		removeRepeatedData();
		
		//mlpack::data::Save(file,m_mFileData);
		return true;
	}
	else
	{
		mlpack::Log::Fatal << "can't read file!" << std::endl;
		return false;
	}
}


void CDataReader::removeRepeatedData()
{
	arma::mat tempMat;
	arma::mat transMat = trans(m_mFileData);
	//int i = 0;
	int start = 0;
	while (start< (transMat.n_rows-1))
	{	
		while (compareVec(transMat.row(start),transMat.row(start+1)) )
		{
			 transMat.shed_row(start+1);
			//  i ++;
			 if (start == (transMat.n_rows -1))
			 {
				 break;
			 }
		
		}
		//tempMat.insert(tempMat.n_rows,m_mFileData.row(start) );
		start ++;

	//	i = start;
	}
	m_mFileData.clear();
	m_mFileData = trans(transMat);
	/*for (int i = 0; i < m_mFileData.n_rows; i ++)
	{
		if (m_mFileData.row(i) == m_mFileData.row(i+1))
		{

		}
	}*/
}

int CDataReader::compareVec(arma::rowvec _vec1,arma::rowvec _vec2)
{
//	int result = 1;
	for (int i = 0; i < _vec1.n_cols; i ++)
	{
		if (_vec1(i) != _vec2(i))
		{
	//		result = 0;
			return 0;
		}
	}
	return 1;
}