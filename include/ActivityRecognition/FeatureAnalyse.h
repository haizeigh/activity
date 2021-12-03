#pragma once
#include "globaltype.h"
#include "mlpack/core.hpp"
#include "mlpack/methods/kernel_pca/kernel_pca.hpp"
#include "mlpack/methods/pca/pca.hpp"
#include <mlpack/methods/pca/decomposition_policies/exact_svd_method.hpp>
#include <mlpack/methods/pca/decomposition_policies/quic_svd_method.hpp>
#include <mlpack/methods/pca/decomposition_policies/randomized_svd_method.hpp>
using namespace mlpack::kernel;
using namespace mlpack::util;
//declaration namespace
using namespace mlpack;
// using namespace mlpack::kpca;
using namespace mlpack::kernel; 
using namespace mlpack::pca;

//! @brief The engine class for dimensional reduction or subspace transformation
/*  
*/
class FeatureAnalyse
{
public:
	FeatureAnalyse();
	~FeatureAnalyse();
	//void create(const std::string& detectorType);
	virtual void apply() = 0;
	//virtual array<array<double>^>^ transform(array<array<double>^>^ arr);
	//virtual array<array<double>^>^ transform(array<array<double>^>^ arr,int dimension);
protected:
	arma::mat m_inputMat;
	int       m_dimensionNum;
	arma::mat m_transformedMat;
	arma::vec m_eigVal;
public:
	arma::mat getTransformationFeats(){return m_inputMat;}
};

/** PCA subclass inherited from class FeatureAnalyse
 *
 */
class PCAFeatureAnalyse : public FeatureAnalyse
{
public:
	PCAFeatureAnalyse(const arma::mat &data, int newDimension);
	PCAFeatureAnalyse(const arma::mat data);
	~PCAFeatureAnalyse();
	void apply();
private:
	PCA<RandomizedSVDPolicy>* m_pcaEngine;

};

