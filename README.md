# Sumbul NMF
Non-negative Matrix Factorization Implementation

Possible error metrics:
- Frobenius Norm (E)
- Kullback-Leibler divergence (KL)
- Itakura-Saito divergence (IS)

Possible initializations for W and H matrices:
- Each elements are randomly selected from a range [0.0, 1.0)
- Random Acol method which initiates each column of the matrix W by averaging p random columns of V
- Selecting mininum error result from multiple randomly initiated runs

For the notation used in the code, you can read the "Finding Underlying Trends and Clustering of Time Series with Non-negative Matrix Factorization and K-Means" [project in my web-site](http://gencersumbul.bilkent.edu.tr/project/time_series_clustering/).
