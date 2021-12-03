#include "FeatureAnalyse.h"

FeatureAnalyse::FeatureAnalyse()
{

}

FeatureAnalyse::~FeatureAnalyse()
{
}


//Class PCAFeatureAnalyse implementation
PCAFeatureAnalyse::PCAFeatureAnalyse(const arma::mat _input)
{
	m_inputMat = _input;
	m_pcaEngine = new PCA<RandomizedSVDPolicy>();
}
PCAFeatureAnalyse::PCAFeatureAnalyse(const arma::mat& _input,int _dim)
{
	m_inputMat = _input;
	m_dimensionNum = _dim;
	m_pcaEngine = new PCA<RandomizedSVDPolicy>();
}
PCAFeatureAnalyse::~PCAFeatureAnalyse()
{
	delete m_pcaEngine;
}
/*
PCAά���������ݾ�����Ϊ����ά������Ϊ���ݵ���
*/
void PCAFeatureAnalyse::apply()
{
	//arma::mat _mat = trans(m_inputMat);
	m_pcaEngine->Apply(m_inputMat,  m_dimensionNum);
	//m_transformedMat = _mat;
	
}

