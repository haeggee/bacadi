# Documentation


## General package structure
The following figure shows how the package and its algorithms are structured


```
cdt package
|
|- independence
|  |- skeleton (Infering the skeleton from data, and removing spurious connections)
|  |  |- Lasso variants (Randomized Lasso[1], Glasso[2], HSICLasso[3])
|  |  |- FSGNN (CGNN variant for feature selection)
|  |  |- Network deconvolution[4]
|  |  |- Skeleton recovery using feature selection algorithms (RFECV[5], LinearSVR[6], RRelief[7], ARD[8,9], DecisionTree)
|  |- stats (pairwise methods for dependency)
|     |- Correlation (Pearson, Spearman, KendallTau)
|     |- Kernel based (NormalizedHSIC[10])
|     |- Mutual information based (MIRegression, Adjusted Mutual Information[11], Normalized mutual information[11])
|
|- generators
|  |- RandomGraphFromData (Generate a random graph similar to inputdata)
|  |- RandomGraphGenerator (Generates a random graph, can generate pairs of variables)
|  |- generate_graph_with_structure (generates a graph with a fixed structure)
|
|- causality
|  |- graph (methods for graph inference)
|  |  |- CGNN[12] method (In tensorflow, pytorch version needs revision)
|  |  |- PC[13]
|  |  |- GES[13]
|  |  |- GIES[13]
|  |  |- LiNGAM[13]
|  |  |- CAM[13]
|  |- pairwise (methods for pairwise inference)
|     |- ANM[14] (Additive Noise Model)
|     |- IGCI[15] (Information Geometric Causal Inference)
|     |- RCC[16] (Randomized Causation Coefficient)
|     |- NCC[17] (Neural Causation Coefficient)
|     |- GNN[12] (Generative Neural Network -- Part of CGNN )
|     |- Bivariate fit (Baseline method of regression)
|     |- GPI[18], PNL[19], Jarfo[20] to implement
|
|- utils
   |- Settings -> CGNN_SETTINGS, SETTINGS (hardware settings)
   |- Loss -> MMD loss [21, 22] & various other loss functions
   |- metrics -> Implements the metrics for graph scoring
   |- Formats -> for importing data formats
   |- Graph -> defines the DirectedGraph and UndirectedGraph class (see below)


```

## Hardware and algorithm settings
The toolbox has a SETTINGS class that defines the hardware settings. Those settings are unique and their default parameters are defined in **_cdt/utils/Settings_**.

These parameters are accessible and overridable via accessing the class:

```python
import cdt
cdt.SETTINGS
```

Moreover, the hardware parameters are detected and defined automatically (including number of GPUs, CPUs, available optional packages) at the **import** of the package using the **cdt.utils.Settings.autoset_settings** method, run at startup.

## The graph class
The whole package revolves around using the **DiGraph** and **Graph** classes from the **networkx** package.

## References

- [1] Wang, S., Nan, B., Rosset, S., & Zhu, J. (2011). Random lasso. The annals of applied statistics, 5(1), 468.
- [2] Friedman, J., Hastie, T., & Tibshirani, R. (2008). Sparse inverse covariance estimation with the graphical lasso. Biostatistics, 9(3), 432-441.
- [3] Yamada, M., Jitkrittum, W., Sigal, L., Xing, E. P., & Sugiyama, M. (2014). High-dimensional feature selection by feature-wise kernelized lasso. Neural computation, 26(1), 185-207.
- [4] Feizi, S., Marbach, D., M??dard, M., & Kellis, M. (2013). Network deconvolution as a general method to distinguish direct dependencies in networks. Nature biotechnology, 31(8), 726-733.
- [5] Guyon, I., Weston, J., Barnhill, S., & Vapnik, V. (2002). Gene selection for cancer classification using support vector machines. Machine learning, 46(1), 389-422.
- [6] Vapnik, V., Golowich, S. E., & Smola, A. J. (1997). Support vector method for function approximation, regression estimation and signal processing. In Advances in neural information processing systems (pp. 281-287).  
- [7] Kira, K., & Rendell, L. A. (1992, July). The feature selection problem: Traditional methods and a new algorithm. In Aaai (Vol. 2, pp. 129-134).
- [8] MacKay,  D.  J.  (1992). Bayesian interpolation. Neural Computation, 4, 415???447.
- [9] Neal, R. M. (1996). Bayesian learning for neural networks. No. 118 in Lecture Notes in Statistics. New York: Springer.
- [10] Gretton, A., Bousquet, O., Smola, A., & Scholkopf, B. (2005, October). Measuring statistical dependence with Hilbert-Schmidt norms. In ALT (Vol. 16, pp. 63-78).
- [11] Vinh, N. X., Epps, J., & Bailey, J. (2010). Information theoretic measures for clusterings comparison: Variants, properties, normalization and correction for chance. Journal of Machine Learning Research, 11(Oct), 2837-2854.
- [12] Goudet, O., Kalainathan, D., Caillou, P., Lopez-Paz, D., Guyon, I., Sebag, M., ... & Tubaro, P. (2017). Learning functional causal models with generative neural networks. arXiv preprint arXiv:1709.05321.
- [13] Spirtes, P., Glymour, C., Scheines, R. (2000). Causation, Prediction, and Search. MIT press.  
- [14] Hoyer, P. O., Janzing, D., Mooij, J. M., Peters, J., & Sch??lkopf, B. (2009). Nonlinear causal discovery with additive noise models. In Advances in neural information processing systems (pp. 689-696).
- [15] Janzing, D., Mooij, J., Zhang, K., Lemeire, J., Zscheischler, J., Daniu??is, P., ... & Sch??lkopf, B. (2012). Information-geometric approach to inferring causal directions. Artificial Intelligence, 182, 1-31.
- [16] Lopez-Paz, D., Muandet, K., Sch??lkopf, B., & Tolstikhin, I. (2015, June). Towards a learning theory of cause-effect inference. In International Conference on Machine Learning (pp. 1452-1461).  
- [17] Lopez-Paz, D., Nishihara, R., Chintala, S., Sch??lkopf, B., & Bottou, L. (2017, July). Discovering causal signals in images. In Proceedings of CVPR.  
- [18] Stegle, O., Janzing, D., Zhang, K., Mooij, J. M., & Sch??lkopf, B. (2010). Probabilistic latent variable models for distinguishing between cause and effect. In Advances in Neural Information Processing Systems (pp. 1687-1695).
- [19] Zhang, K., & Hyv??rinen, A. (2009, June). On the identifiability of the post-nonlinear causal model. In Proceedings of the twenty-fifth conference on uncertainty in artificial intelligence (pp. 647-655). AUAI Press.
- [20] Fonollosa, J. A. (2016). Conditional distribution variability measures for causality detection. arXiv preprint arXiv:1601.06680.
- [21] Gretton, A., Borgwardt, K. M., Rasch, M. J., Sch??lkopf, B., & Smola, A. (2012). A kernel two-sample test. Journal of Machine Learning Research, 13(Mar), 723-773.
- [22] Li, Y., Swersky, K., & Zemel, R. (2015). Generative moment matching networks. In Proceedings of the 32nd International Conference on Machine Learning (ICML-15) (pp. 1718-1727).  
