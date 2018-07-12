from __future__ import division

import pandas as pd
from numpy import sum, unique
from ut.ml.synthetic.x_and_y import make_multimodal_blobs


class ModelTester(object):
    def __init__(self, model_class, get_data=make_multimodal_blobs, include_model_attrs=()):
        self.model_class = model_class
        self.get_data = get_data
        self.include_model_attrs = include_model_attrs

    def test_model_against_data(self, model, X, y):
        d = {
            'npts': X.shape[0],
            'ndims': X.shape[1],
            'n_unik_y': len(unique(y))}
        try:
            model_attrs = set(vars(model))
            for attr in self.include_model_attrs:
                if attr in model_attrs:
                    d['model' + '_' + attr] = getattr(model, attr)

            model.fit(X, y)

            if hasattr(model, 'transform'):
                d['transform_shape'] = model.transform(X).shape
            if hasattr(model, 'predict'):
                d['accuracy'] = sum(model.predict(X) == y) / float(len(y))

            model_attrs = set(vars(model))
            for attr in self.include_model_attrs:
                if attr in model_attrs:
                    d['model' + '_' + attr] = getattr(model, attr)

        except Exception as e:
            d['error'] = True
            d['msg'] = e.args[0]
        return d

    def test_data_on_several_models(self, X, y, model_kwargs_list=None):
        if model_kwargs_list is None:
            model_kwargs_list = [None]
        d = list()
        for model_kwargs in model_kwargs_list:
            model = self.model_class(**model_kwargs)
            d.append(self.test_model_against_data(model, X, y))
        return d

    def test_model_on_several_datas(self, model, data_kwargs_list=None):
        if data_kwargs_list is None:
            data_kwargs_list = [None]
        d = list()
        for data_kwargs in data_kwargs_list:
            X, y = self.get_data(**data_kwargs)
            d.append(self.test_model_against_data(model, X, y))
        return d

    def test_several_models_and_datas(self, model_kwargs_list=None, data_kwargs_list=None, output='dict'):
        if model_kwargs_list is None:
            model_kwargs_list = [None]
        if data_kwargs_list is None:
            data_kwargs_list = [None]
        d = list()
        for data_kwargs in data_kwargs_list:
            X, y = self.get_data(**data_kwargs)
            d.extend(self.test_data_on_several_models(X, y, model_kwargs_list))
        if output == 'df':
            d = self.df_of(d)
        return d

    @staticmethod
    def df_of(d):
        df = pd.DataFrame(d)
        if 'error' in df.columns:
            df['error'] = df['error'].fillna(False)
        df = df.fillna('')
        return df
