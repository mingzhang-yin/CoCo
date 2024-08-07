# Optimization-based Causal Estimation from Heterogeneous Environments
This repository contains the code to implement examples of CoCo as described in [Optimization-based Causal Estimation from Heterogenous Environments](https://www.jmlr.org/papers/volume25/21-1028/21-1028.pdf) by Mingzhang Yin, Yixin Wang and David Blei.


# Abstract

This paper presents an optimization approach to causal estimation. Given data that contains covariates and an outcome, 
which covariates are causes of the outcome, and what is the strength of the causality? In classical machine learning, 
the goal of optimization is to maximize predictive accuracy. However, some covariates might exhibit non-causal association
with the outcome. Such spurious associations provide predictive power for classical ML, but they prevent us from interpreting
the result causally.  This paper proposes CoCo, an optimization algorithm that bridges the gap between pure prediction and 
causal inference. CoCo leverages the recently-proposed idea of environments, datasets of covariates/response where the causal
relationships remain invariant but where the distribution of the covariates changes from environment to environment. Given datasets from multiple environments—and ones that exhibit sufficient
heterogeneity—CoCo maximizes an objective for which the only solution is the causal
solution. We describe the theoretical foundations of this approach and demonstrate its
effectiveness on simulated and real datasets. Compared to classical ML and existing methods,
CoCo provides more accurate estimates of the causal model and more accurate predictions under interventions.
# Paper Repository

* The python scripts are included in the folder `src/`. The bash files to submit jobs to the cluster are included as `run.sh` (note that this depends on the machine and the bash file syntax may need to be changed accordingly).

* To generate data for wild animal classification, follow the steps described in [this repo](https://github.com/fastforwardlabs/causality-for-ml). Other data can be self-generated or auto-downloaded. 

* To run the code
```
cd ./src/exp_xxx
chmod +x run.sh
bash run.sh
```

* Below is the bibliography information of the paper:
```
@article{yin2024optimization,
  title={Optimization-based Causal Estimation from Heterogeneous Environments},
  author={Yin, Mingzhang and Wang, Yixin and Blei, David M},
  journal={Journal of Machine Learning Research},
  volume={25},
  pages={1--44},
  year={2024}
}
```
