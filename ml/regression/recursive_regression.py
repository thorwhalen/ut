from sklearn.base import RegressorMixin, TransformerMixin
from sklearn.linear_model import LinearRegression
from numpy import zeros, vstack


class RecursiveRegressionTransformer(RegressorMixin, TransformerMixin):
    def __init__(self, model_class=LinearRegression, n_cycles=2, **model_kwargs):
        self.model_class = model_class
        self.n_cycles = n_cycles
        self.model_kwargs = model_kwargs

    def fit(self, X, y, **kwargs):
        y = y.copy()  # because we're going to change y in here!
        self.models_ = list()
        for i in range(self.n_cycles):
            if i != 0:
                y -= model.predict(X)  # replace y by residues of last model computed
            model = self.model_class(**self.model_kwargs).fit(X, y, **kwargs)
            self.models_.append(model)
        return self

    def predict(self, X):
        preds = zeros(len(X))
        for model in self.models_:
            preds += model.predict(X)
        return preds

    def transform(self, X):
        transformed = list()
        for i, model in enumerate(self.models_):
            transformed.append(model.predict(X))
        return vstack(transformed).T
