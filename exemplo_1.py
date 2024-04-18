import matplotlib
import matplotlib.pyplot as plt
#pip install matplotlib

from sklearn.datasets import make_regression
#pip install -U scikit-learn

x, y = make_regression(n_samples=200, n_features=1, noise=30)

plt.scatter(x,y)
plt.show()