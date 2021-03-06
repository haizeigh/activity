#include "DataProc.h"


DataProc::DataProc()
{

}
DataProc::~DataProc()
{

}
DataProc::DataProc(parameters _param)
{
	m_uiWinWidth = _param.uiSlideWinWidth;
	m_uiWinStep  = _param.uiSlideWinStep;
}

//#ifdef _FILTER_MOVING_SMOOTHING_
void DataProc::filterMV(arma::mat fileMat)
{
	/*for (int i = 0; i < dataArray.cols(); ++i)
	{
      if (i < 1)
	  {
		  m_mFilter.row(0)(i) = dataArray.row(0)(i);
		  m_mFilter.row(1)(i) = dataArray.row(1)(i);
		  m_mFilter.row(2)(i) = dataArray.row(2)(i);
	  }
	  else
	  {
		  m_mFilter.row(0)(i) = (dataArray.row(0)(i -1) + dataArray.row(0)(i) +  dataArray.row(0)(i + 1)) / 3;
		  m_mFilter.row(1)(i) = (dataArray.row(1)(i -1) + dataArray.row(1)(i) +  dataArray.row(1)(i + 1)) / 3;
		  m_mFilter.row(2)(i) = (dataArray.row(2)(i -1) + dataArray.row(2)(i) +  dataArray.row(2)(i + 1)) / 3;
	  }
	}*/

    for (int i = 1; i < fileMat.n_cols - 1; ++i){

        for (int j = 0; j < fileMat.n_rows; ++j) {
            fileMat[j,i] = (fileMat[j,i-1] + fileMat[j,i] + fileMat[j,i+1]) /3 ;
        }

    }


}
//#endif //_FILTER_MOVING_SMOOTHING_

BOOL8 DataProc::segment()
{
	UINT32 uiDataLength = m_mRawDataStream.n_cols;
	UINT32 uiCurrPos    = 0;
	UINT32 uiIndex      = 0;

	if (uiDataLength > m_uiWinWidth)
	{
		while (uiCurrPos < uiDataLength - m_uiWinWidth)
		{
			m_cSegmentedDataSlices.resize(ACC_AXIS_NUM,m_uiWinWidth,uiIndex+1);
			mat b= m_mRawDataStream.submat(span::all,span(uiCurrPos,m_uiWinWidth-1+uiCurrPos));
			m_cSegmentedDataSlices.slice(uiIndex) = m_mRawDataStream.submat(span::all,span(uiCurrPos,m_uiWinWidth-1+uiCurrPos));
			uiIndex ++;
			uiCurrPos += m_uiWinStep;
		}
		return true;
	}
	else
	{
		//mlpack::Log::Warn<<"The data file is too short, can't be segmented!!!"<<std::endl;
		return false;
	}

}

void DataProc::concatExtractedFeatureToMat(arma::vec _extractedFeature)
{
	//m_mExtractedFeature.insert_cols(0,_extractedFeature); ///always insert column vector at column 0
	m_mExtractedFeature.insert_cols(m_mExtractedFeature.n_cols,_extractedFeature);
}

arma::mat DataProc::getExtractedFeatures()
{
	//normalization();
	//computeFeatureScale(m_mExtractedFeature);
	//normalization();
	return m_mExtractedFeature;
}
//????????????????????????????????????????????????????
//??????????(0~1)????????????????????????????????????????????-1~1??????
//norm = ((x-min)/(max-min))*2 -1
void DataProc::normalization()
{
	//between 0~1
	arma::rowvec _vec = arma::max(m_mExtractedFeature,0)-arma::min(m_mExtractedFeature,0);
	m_mExtractedFeature.each_row() -= arma::min(m_mExtractedFeature,0);
	m_mExtractedFeature.each_row() /= _vec;

	//between -1~1
	arma::rowvec vec_(m_mExtractedFeature.n_cols);
	vec_.fill(2);
	//arma::mat temp = m_mExtractedFeature;
	m_mExtractedFeature *=2;
//	m_mExtractedFeature.each_row() vec_;
	vec_.fill(1);
//	m_mExtractedFeature.each_row() -= vec_ ;
	m_mExtractedFeature -= 1;
}

void DataProc::computeFeatureScale(arma::mat &_mat , bool _SAVE,em_featurescale_type _type)
{
	
	arma::mat featureMean;
	arma::mat featureStd;
	arma::mat featureMin;
	arma::mat featureMax;
	if (_SAVE == true)
	{
		switch (_type)
		{
		case EM_FEATURESCALE_MINMAXSCALER:
			featureMin.insert_cols(0,arma::min(_mat,1));
			featureMax.insert_cols(0,arma::max(_mat,1));
			featureMin.save("FeatureMin.mat");
			featureMax.save("FeatureMax.mat");
			break;
		case EM_FEATURESCALE_STANDARDSCALER:
			featureMean.insert_cols(0,arma::mean(_mat,1));
			featureStd.insert_cols(0, arma::stddev(_mat,0,1));
			featureMean.save("FeatureMean.mat");
			featureStd.save("FeatureStd.mat");
			break;
		default:
			break;
		}
	}
	else
	{
		switch (_type)
		{
		case  EM_FEATURESCALE_MINMAXSCALER:
			featureMin.load("FeatureMin.mat");
			featureMax.load("FeatureMax.mat");
			break;
		case EM_FEATURESCALE_STANDARDSCALER:
			featureMean.load("FeatureMean.mat");
			featureStd.load("FeatureStd.mat");
			break;
		default:
			break;

		}
	}
	arma::vec _vec;
	arma::uvec ind;
	switch (_type)
	{
	case EM_FEATURESCALE_MINMAXSCALER: 
		_vec = featureMax - featureMin;
         ind = arma::find(_vec == 0);
		_vec(ind) = _vec(ind) + 1;
		_mat.each_col() -= featureMin;
		_mat.each_col() /= _vec;
		_mat *=2;
		_mat -= 1;
		break;
	case EM_FEATURESCALE_STANDARDSCALER:
		_mat.each_col()                -= featureMean.col(0);
		arma::vec temp = featureStd.col(0);
        ind = arma::find(temp == 0);
		temp(ind) = temp(ind) + 1;
		_mat.each_col()                /= temp;
		break;
	}


	//if (SAVE == true)
	//{
	//	featureMean.insert_cols(0,arma::mean(_mat,1));
	//	featureStd.insert_cols(0, arma::stddev(_mat,0,1));
	//	featureMean.save("FeatureMean.mat");
	//	featureStd.save("FeatureStd.mat");
	//}
	//else
	//{
	//	featureMean.load("FeatureMean.mat");
	//	featureStd.load("FeatureStd.mat");
	//}
	//_mat.each_col()                -= featureMean.col(0);
	//arma::vec temp = featureStd.col(0);
	//arma::uvec ind = arma::find(temp == 0);
	//temp(ind) = 1;
	//_mat.each_col()                /= temp;
	////_mat.row(ind) = _mat.row(ind) + 0.1;
	////_


	////between -1~1
	//arma::vec _min = arma::min(_mat,1);
	//arma::vec _max = arma::max(_mat,1);

	//arma::vec _vec = _max - _min;
	//arma::uvec ind = arma::find(_vec == 0);
	//_vec(ind) = _vec(ind) + 0.1;
	//_mat.each_col() -= _min;
	//_mat.each_col() /= _vec;

	//_mat *=2;
	//_mat -= 1;*/
}

std::string DataProc::int2String(int _type)
{
	std::string str;
	switch (_type)
	{
	case 0:
		str = "walking";
		break;
	case 1:
		str = "running";
		break;
	case 2:
		str = "sitting";
		break;
	case 3:
		str = "lying";
		break;
	case 4:
		str = "standing";
		break;
	default:
		str = "";
		break;
	}
	return str;
}

