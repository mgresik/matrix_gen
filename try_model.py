from pyod.models.ecod import ECOD
from sklearn.decomposition import PCA

pca = PCA(n_components=100)
X_small = pca.fit_transform(X)

# Fast search of anomalies
clf = ECOD()
y_pred = clf.fit_predict(X_small)