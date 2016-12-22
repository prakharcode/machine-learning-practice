from __future__ import division
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures as pf

X_train = [[6],[8],[10],[14],[18]]
y_train = [[7],[9],[13],[17.5],[18]]
X_test = [[6],[8],[11],[16]]
y_test = [[8],[12],[15],[18]]

lr = LinearRegression()

lr.fit(X_train, y_train)
xx = np.linspace(0,26,100)
yy = lr.predict(xx.reshape(xx.shape[0],1))
plt.plot(xx,yy)
print 'Simple linear regression r-squared',lr.score(X_test,y_test)

quadratic_featurizer = pf(degree = 3)
X_train_quadratic = quadratic_featurizer.fit_transform(X_train)
X_test_quadratic = quadratic_featurizer.transform(X_test)

lr_quadratic = lr
lr_quadratic.fit(X_train_quadratic, y_train)
xx_quadratic = quadratic_featurizer.transform(xx.reshape(xx.shape[0],1))

plt.plot(xx,lr_quadratic.predict(xx_quadratic), c='r',linestyle='--')
plt.title('Pizza price regressed on diameter')
plt.xlabel('Diameter in inches')
plt.ylabel('Price in dollars')
plt.axis([0, 25, 0, 25])
plt.grid(True)
plt.scatter(X_train, y_train)
plt.show()

print X_train
print X_train_quadratic
print X_test
print X_test_quadratic
print y_test
print 'Quadratic regression r-squared',lr_quadratic.score(X_test_quadratic,y_test)

