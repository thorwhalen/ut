__author__ = 'thor'


from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
import prettyplotlib as ppl


def xy_linear_regression(df, x_var, y_var, **kwargs):
    fit_intercept = kwargs.get('fit_intercept', True)
    d = df[[x_var, y_var]]
    d = d.dropna()

    x = np.matrix(d[x_var]).transpose()
    y = np.matrix(d[y_var]).transpose()

    clf = linear_model.LinearRegression(
        fit_intercept=kwargs.get('fit_intercept', True),
        normalize=kwargs.get('normalize', False))
    clf.fit(x, y)
    intercept, slope = clf.intercept_[0], clf.coef_[0][0]

    if kwargs.get('plot', False):
        ppl.scatter(x=d[x_var], y=d[y_var]);
        x_line = np.array([0, d[x_var].max()])
        plt.plot(x_line, intercept + slope * x_line, 'k-')
        plt.annotate('slope={slope:.4f}\nintercept={intercept:.4f}'.format(slope=slope, intercept=intercept),
                     (0.05, 0.9), xycoords='axes fraction')
        plt.xlabel(x_var)
        plt.ylabel(y_var)

    return intercept, slope


