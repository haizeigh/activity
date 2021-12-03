
#include <iostream>
#include <filesystem>

#include "mlpack/core.hpp"
#include "globaltype.h"
#include <sys/types.h>
#include <dirent.h>
#include <sys/stat.h>

using namespace mlpack;

 /**
  * This class is used to access txt files.
  * 
  * The goal is to collect the train data for activity recognition
  */
class CDataReader
{
public:
	/** @brief constructor for class CDataReader
	*  
	*/
	CDataReader();
	
	/** @brief read a directory to obtain the name list of subdirectory or files
	*   @param dirName the name of the directory ready to read
	*   @return the status whether the directory is successful to read or not
	*/
	BOOL8 readDir(std::string dirName);

	/** @brief read a file and save as arma::mat format
	*   @param fileName the file name
	*   @return the status whether the file is successful to read or not
	*/
	BOOL8 readFile(const std::string fileName);

	/** @brief get the base train directory
	*/
	inline std::vector<std::string> getDirsList(){return m_vDirNames;}

	inline std::vector<std::string> getFilesList(){return m_vFileNames;}

	inline arma::mat getFileData(){return m_mFileData;}

private:
	std::vector<std::string> m_vFileNames;          /// the name list for special posture raw data files 
	std::vector<std::string> m_vDirNames;           /// sub-directory lists for storing the name of different postures

	arma::mat m_mFileData;                     /// matrix for storing the data from a train file

private:

	/**
	 *    @brief remove repeated lines in original data files
	 */
	void removeRepeatedData();

	int compareVec(arma::rowvec _vec1,arma::rowvec _vec2);

};

