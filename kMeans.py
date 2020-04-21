import numpy as np
import sklearn
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn import metrics


class example1:
    def __init__(self):
        self.digits = load_digits()
        self.data = scale(self.digits.data)
        self.y = self.digits.target
        self.k = 10
        self.samples, self.features = self.data.shape

    def main(self):
        clf = KMeans(n_clusters=self.k, init="random", n_init=10)
        self.bench_k_means(clf, "1", self.data)

    def bench_k_means(self, estimator, name, data):
        estimator.fit(data)
        print('%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
              % (name, estimator.inertia_,
                 metrics.homogeneity_score(self.y, estimator.labels_),
                 metrics.completeness_score(self.y, estimator.labels_),
                 metrics.v_measure_score(self.y, estimator.labels_),
                 metrics.adjusted_rand_score(self.y, estimator.labels_),
                 metrics.adjusted_mutual_info_score(self.y, estimator.labels_),
                 metrics.silhouette_score(data, estimator.labels_,
                                          metric='euclidean')))


if __name__ == '__main__':
    example1().main()