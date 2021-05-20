# Optimization-based Causal Estimation from Heterogenous Environments
* Implemention of CoCo as described in [Optimization-based Causal Estimation]() by Mingzhang Yin, Yixin Wang and David Blei.

This paper presents an optimization approach to causal
  estimation. Given data that contains covariates and an outcome,
  which covariates are causes of the outcome, and what is the strength
  of the causality? In classical machine learning, the goal of
  optimization is to maximize predictive accuracy. However, some
  covariates might exhibit non-causal association to the outcome. Such
  spurious associations provide predictive power for classical ML, but
  they prevent us from interpreting the result causally.  This paper
  proposes CoCo, an optimization algorithm that bridges the gap
  between pure prediction and causal inference. CoCo leverages
  the recently-proposed idea of environments, datasets of
  covariates/response where the causal relationships remain invariant
  but where the distribution of the covariates changes from
  environment to environment. Given datasets from multiple
  environments---and ones that exhibit sufficient
  heterogeneity---CoCo maximizes an objective for which the only
  solution is the causal solution. We describe the theoretical
  foundations of this approach and demonstrate its effectiveness on
  simulated and real datasets. Compared to classical ML and the
  recently-proposed IRMv1, CoCo provides more accurate estimates
  of the causal model.

Below is the paper to cite if you find the algorithms in this repository useful in your own research:
```
```

* To generate data for wild animal classification, follow the steps describe by [Cloudera](https://github.com/fastforwardlabs/causality-for-ml). 

* To run the code
```
cd ./src/exp_xxx
chmod +x run.sh
bash run.sh
```
