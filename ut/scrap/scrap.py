from functools import partial
import numpy as np
from hear.tools import AudioSegments
from omodel.gen_utils.chunker import fixed_step_chunker
from collections import defaultdict
import abc
import pandas as pd
from omodel.outliers.ui_score_function import tune_ui_map, make_ui_score_mapping


# -----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------AnnotsPartitions---------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------


class AnnotsPartitions(metaclass=abc.ABCMeta):
    """
    Interface to manipulate annotations in a consistent manner, whether they live in a pandas dataframe, mongo db...
    """

    # TODO: Consider replacing sref, rel_ts by functions
    @abc.abstractmethod
    def __init__(self, annots, sref_name, rel_ts_name):
        self.annots = annots
        self.sref_name = sref_name
        self.rel_ts_name = rel_ts_name

    @abc.abstractmethod
    def index_to_row(self, idx):
        """Given a integer, the method retrieve the annotation in that row number"""
        raise NotImplementedError

    @abc.abstractmethod
    def get_annot_field(self, annot, field: str):
        """Given a string field, the method retrieve value of the annotation for that field"""
        raise NotImplementedError

    @abc.abstractmethod
    def get_col(self, field):
        """Given a string field, the method retrieve value of the annotation for that field"""
        raise NotImplementedError

    @abc.abstractmethod
    def set_annot_column(self, field, vals):
        """
        Given a string field, the method set the value of the annotations for that field, like setting
        a column in pandas
        """
        raise NotImplementedError

    @abc.abstractmethod
    def sort(self, sort_col=None):
        """
        Sort the annotations in order of value in sort_col
        """
        if sort_col is None:
            sort_col = self.rel_ts_name
        raise NotImplementedError

    @abc.abstractmethod
    def add_index(self):
        """
        Add or replace the row index in annotations
        """
        raise NotImplementedError

    # TODO: Do we need/want this method here?
    @abc.abstractmethod
    def select_from_string(self, col_condition):
        """
        Convert a UI input into a selection for the annotations. For example if annotations is a pandas df and
        the string is "rpm > 200", this methods should do something like that:
        return annotation[annotations['rpm'] > 200]
        """
        raise NotImplementedError

    @abc.abstractmethod
    def select_from_strings(self, col_conditions):
        """
        Convert a UI input into a selection for the annotations. For example if annotations is a pandas df and
        the string is "rpm > 200", this methods should do something like that:
        return annotation[annotations['rpm'] > 200]
        """
        raise NotImplementedError

    @abc.abstractmethod
    def group_from_cols(self, cols):
        """
        Implement a grouby method, the results should be a dictionary like object with key being the existing
        combination of values in cols and the values the list of corresponding indices
        """
        raise NotImplementedError

    @abc.abstractmethod
    def create_col_from_val_list(self, vals, col_name, val_func=lambda x: x):
        """
        Method to add a column for annotations, where the values in the columns are the image of those in vals
        as given by val_func
        """
        raise NotImplementedError

    @abc.abstractmethod
    def find_blocks_from_col(self, col, tol=0):
        """
        Find block of consecutive values which are close enough in value in a column (according to tol)
        The result is a list of the form [0, 0, 1, 1, 1, 2, 3, 3...]
        """
        raise NotImplementedError

    @abc.abstractmethod
    def find_blocks_from_index(self):
        """
        Find block of consecutive indices
        """
        raise NotImplementedError

    @abc.abstractmethod
    def add_blocks_col(self, col, new_col=None):
        """Add a new column of name new_col (or a default) with values given by the find_blocks method"""
        raise NotImplementedError


# -----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------Example of AnnotsPartitions----------------------------------------
# -----------------------------------------------------------------------------------------------------------------------


class PandasAnnotsPartition(AnnotsPartitions):
    """
    A special case of AnnotsPartitions, for pandas dataframe (and any annotations that can be turned into pandas a df)
    """

    def __init__(self, annots, sref_name='filename', rel_ts_name='bt'):
        self.annots = annots
        self.sref_name = sref_name
        self.rel_ts_name = rel_ts_name

    def index_to_row(self, idx):
        return self.annots.loc[idx]

    def get_annot_field(self, annot, field):
        return annot[field]

    def get_col(self, field):
        return self.annots[field]

    def set_annot_column(self, field, vals):
        self.annots[field] = vals

    def sort(self, sort_cols=None):
        if sort_cols is None:
            sort_cols = [self.sref_name, self.rel_ts_name]
        self.annots.sort_values(sort_cols, inplace=True)

    def add_index(self):
        self.annots.index = range(len(self.annots))

    def select_from_string(self, col_condition):
        """Select from the string col_condition, for example "20 <= rpm < 200" """
        return self.annots.eval(col_condition)

    def select_from_strings(self, col_conditions):
        """
        Not safe, per pandas doc about eval:
          "This allows eval to run arbitrary code, which can make you vulnerable to code injection
           if you pass user input to this function."
        """
        df_select = self.annots
        for col_cond in col_conditions:
            df_select = df_select.eval(col_cond)
        return df_select

    def group_from_cols(self, cols):
        'The last column is intended to be train/test (train=1) the others are partitioning the data into model target'
        self.groups_ = tuple(cols)
        return self.annots.groupby(cols)

    def create_col_from_val_list(self, vals, col_name, val_func=lambda x: x):
        self.set_annot_column(field=col_name, vals=[val_func(val) for val in vals])

    def find_blocks_from_col(self, col, tol=0):
        return self.annots[col].diff().gt(tol).cumsum()

    def find_blocks_from_index(self):
        return list(pd.Series(self.annots.index).diff().gt(1).cumsum())

    def add_blocks_col(self, col, new_col=None):
        if new_col is None:
            new_col = col + '_block'
        self.set_annot_column(new_col, self.find_blocks_from_col(col))


# TODO: move the code below somewhere else
# -----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------DACC with AnnotationPart instance----------------------------------
# -----------------------------------------------------------------------------------------------------------------------


DFLT_SR = 51200
DFLT_MAX_ANNO_DURATION_SEC = 1
DFLT_INDEX_TO_SECOND_SCALE = 1e-6
DFLT_CHK_SIZE = 2048
DFLT_CHK_STEP = 2048
# DFLT_CHUNKER = partial(fixed_step_chunker, chk_size=DFLT_CHK_SIZE, chk_step=DFLT_CHK_STEP)
DFLT_SPECTRA_COMP = lambda x: np.abs(np.fft.rfft(x))


# TODO This class uses a simple regular chunker, could be generalized
# NOTE: we could also use the raw annotation object instead of annot_part but then the init would be more complicated
# as we would need to specify index_to_row and get_annot_field for the specific format of annotation
# The advantage would be that the dacc could be used independently of an instance of annot_part. I decided against
# since annot_part will soon exist for the main annotations format
class Dacc:
    """
    Class to access the data. Wf_store must be a waveform store, annotations a "pandas df" like object. It must
    have two functions index_to_row, get_annot_field, which both needs to provided. annot_to_wf_key is a function
    taking each "row" of annotations to the key of the wf_store and timestamp_name is the field of the annotations
    containing the bt of the annotation
    """

    def __init__(
        self,
        wf_store,
        annot_part,
        sr=DFLT_SR,
        max_annot_duration_sec=DFLT_MAX_ANNO_DURATION_SEC,
        index_to_seconds_scale=DFLT_INDEX_TO_SECOND_SCALE,
        chk_size=DFLT_CHK_SIZE,
        chk_step=DFLT_CHK_STEP,
        fft=DFLT_SPECTRA_COMP,
    ):

        self.wf_store = wf_store
        self.annot_part = annot_part
        # TODO: do we really want to cache the segements by default? Seems memory wasteful most of the time
        self.audio_seg = AudioSegments(
            src_to_wfsr=lambda x: (wf_store[x], sr),
            index_to_seconds_scale=index_to_seconds_scale,
        )
        self.index_to_seconds_scale = index_to_seconds_scale
        self.max_annot_duration_sec = max_annot_duration_sec
        self.chk_size = chk_size
        self.chk_step = chk_step
        self.fft = fft

    # Get selected data in the wf_store/annot_store
    # TODO: This could be more efficiently dealt with: instead of looking everytime at the next annot, we can
    #  always return the 1 second default and keep track of where that leave the "fresh" wf at, to avoid covering the
    #  part of the wf twice. This way we divide almost by two the number of self.annot_part.get_annot_field calls
    #  if done that way, we need to take care of potential change of sref!
    def wf_and_annot_for_annot_idx(self, annot_idx):
        """
        Example of use:
        dacc.wf_and_annot_for_annot_id(10)
        """
        annot = self.annot_part.index_to_row(annot_idx)
        bt = self.annot_part.get_annot_field(annot, self.annot_part.rel_ts_name)
        try:
            next_annot = self.annot_part.index_to_row(annot_idx + 1)
            tt = self.annot_part.get_annot_field(
                next_annot, self.annot_part.rel_ts_name
            )
            if tt <= bt:
                tt = bt + self.max_annot_duration_sec / self.index_to_seconds_scale
                next_continuous = False
            elif tt <= bt + self.max_annot_duration_sec / self.index_to_seconds_scale:
                next_continuous = True
            else:
                tt = bt + self.max_annot_duration_sec / self.index_to_seconds_scale
                next_continuous = False
        except:
            tt = bt + self.max_annot_duration_sec / self.index_to_seconds_scale
            next_continuous = False
        wf_key = self.annot_part.get_annot_field(annot, self.annot_part.sref_name)
        return self.audio_seg[wf_key, bt:tt], annot, next_continuous

    def wf_and_annot_for_annot_idxs(self, annot_idx_list):
        """
        Example of use:
        dacc.get_row_and_wf_for_row_keys([10, 11, 12])
        """
        for annot_idx in annot_idx_list:
            yield self.wf_and_annot_for_annot_idx(annot_idx)

    def spectr_for_annot_keys(self, annot_idx_list):
        """
        Example of use:
        dacc.get_row_and_wf_for_row_keys([10, 11, 12, 20, 21, 22, 23])
        """
        buffer = []
        next_continuous = True
        for wf, annot, continuous in self.wf_and_annot_for_annot_idxs(annot_idx_list):
            if next_continuous:
                buffer.extend(wf)
            else:
                buffer = list(wf)
            while len(buffer) >= self.chk_size:
                chk, buffer = buffer[: self.chk_size], buffer[self.chk_step :]
                yield self.fft(chk), annot
            next_continuous = continuous


# TODO: move the code below somewhere else and clean up/generalized for more than anomaly model
# -----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------Training/running code----------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------


from omodel.outliers.outlier_model import OutlierModel
from omodel.outliers.gmm_stroll import GmmStroll
from omodel.fv.chained_spectral_projector import GeneralProjectionLearner
from sklearn.preprocessing import StandardScaler


def train_featurizer(
    X_train, y_train, featurizer_chain=({'type': 'pca', 'args': {'n_components': 20}},)
):
    gpl = GeneralProjectionLearner(chain=featurizer_chain)
    try:
        gpl.fit(X_train, y_train)
    except Exception as E:
        gpl.fit(X_train)
    return gpl


def featurize(X, gpl):
    return gpl.transform(X)


def train_normalizer(fvs_train, normalizer_class=StandardScaler):
    if normalizer_class is None:
        return None
    else:
        normalizer = normalizer_class()
        normalizer.fit(fvs_train)
        return normalizer


def normalize(X, normalizer):
    if normalizer is None:
        return X
    else:
        return normalizer.transform(X)


def train_model(fvs_train, y_train, model_name=GmmStroll, **model_kwargs):
    model_class = get_model_cls(model_name)
    model = model_class(**model_kwargs)
    try:
        model.fit(fvs_train, y_train)
    except Exception as e:
        model.fit(fvs_train)
    return model


def get_model_scores(fvs, model, model_type='anomaly', predict_proba=False):
    if model_type == 'anomaly':
        if 'n_centroids' in model.__dict__:
            scores = np.array([model.get_score(fv)['outlier_score'] for fv in fvs])
        else:
            scores = np.array(model.score_samples(fvs))
        return scores
    elif model_type == 'classification':
        scores = model.predict_proba(fvs)
        predict = model.predict(fvs)
        if predict_proba:
            return np.array(scores)
        else:
            return np.array(predict)


model_for_name = {'Stroll': OutlierModel, 'GmmStroll': GmmStroll}


def get_model_cls(model_name):
    if isinstance(model_name, str):
        model_class = model_for_name.get(model_name, None)
        if model_class is None:
            from sklearn.utils import all_estimators

            estimator_for_name = dict(all_estimators())
            model_class = estimator_for_name.get(model_name, None)
        assert model_class is not None, f"Couldn't find a model for {model_name}"
    else:
        model_class = model_name
    return model_class


def train_pipeline(
    spectra,
    classes=None,
    featurizer_chain=({'type': 'pca', 'args': {'n_components': 20}},),
    normalizer_class=StandardScaler,
    model_class='Stroll',
    train_ui_map=False,
    **model_kwargs,
):
    gpl = train_featurizer(
        X_train=spectra, y_train=classes, featurizer_chain=featurizer_chain
    )
    fvs = featurize(spectra, gpl)

    normalizer = train_normalizer(fvs, normalizer_class=normalizer_class)
    norm_fvs = normalize(fvs, normalizer)
    model = train_model(
        norm_fvs, classes, model_name=model_class, **model_kwargs['model_class_params']
    )
    mod = {'gpl': gpl, 'normalizer': normalizer, 'model': model, 'ui_amp': None}
    if train_ui_map:
        train_scores = run_pipeline(spectra, use_ui_map=False, **mod)
        ui_map_params = tune_ui_map(train_scores)
        ui_map_params = {
            'min_lin_score': np.min(ui_map_params),
            'max_lin_score': np.max(ui_map_params),
            'top_base': ui_map_params[2],
            'bottom_base': ui_map_params[3],
        }
        mod['ui_map_params'] = ui_map_params
    return mod


def run_pipeline(spectra, gpl, normalizer, model, use_ui_map, **params):
    fvs = featurize(spectra, gpl)
    norm_fvs = normalize(fvs, normalizer)
    scores = get_model_scores(norm_fvs, model)
    if use_ui_map:
        ui_map = make_ui_score_mapping(**params['ui_map_params'])
        scores = [ui_map(score) for score in scores]
    return scores


dflt_dacc_params = {
    'sr': DFLT_SR,
    'index_to_seconds_scale': DFLT_INDEX_TO_SECOND_SCALE,
    'max_annot_duration_sec': DFLT_MAX_ANNO_DURATION_SEC,
    'chk_size': DFLT_CHK_SIZE,
    'chk_step': DFLT_CHK_STEP,
    'fft': DFLT_SPECTRA_COMP,
}

dflt_train_params = {
    'featurizer_chain': ({'type': 'pca', 'args': {'n_components': 20}},),
    'normalizer_class': None,
    'model_class': 'GmmStroll',
    'model_class_params': {},
    'train_ui_map': True,
}


class ModelsTrainAndRun:
    def __init__(
        self,
        wf_store,
        annot_part,
        dflt_dacc_params=dflt_dacc_params,
        dflt_train_params=dflt_train_params,
    ):
        """
        Class to manage, train and run serveral models

        :param wf_store: a store containing the waveform
        :param annot_part: an instance of a subclass of AnnotsPartitions
        :param sort: whether or not to sort the annotations in annot_part_inst
        :param dflt_dacc_params: default parameters for the dacc, can be overridden at fitting time
        :param dflt_train_params: default parameters to train a model with, can be overridden at training time
        """

        self.wf_store = wf_store
        self.annot_part = annot_part
        self.dflt_dacc_params = dflt_dacc_params
        self.dflt_train_params = dflt_train_params
        self.model_pipes = dict()
        self.results = dict()
        self.merge_func = dict()

    def train_model(self, indices, dacc_params=None, train_params=None):
        """
        Train a model using the data associated with the annotations with indices in indices
        :param indices: a list of annotations index
        :return: a dict of the form {'gpl': gpl, 'normalizer': normalizer, 'model': model}
        """
        if dacc_params is None:
            dacc_params = self.dflt_dacc_params
        if train_params is None:
            train_params = self.dflt_train_params

        dacc = Dacc(wf_store=self.wf_store, annot_part=self.annot_part, **dacc_params)

        spectra, annots = zip(*dacc.spectr_for_annot_keys(indices))
        spectra = np.array(spectra)
        mod = train_pipeline(spectra, **train_params)
        mod.update(dacc_params)
        mod.update(train_params)
        return mod

    def save_model(self, model_id, mod):
        self.model_pipes[model_id] = mod

    def run_model(self, indices, model_name, dacc_params=None, use_ui_map=True):
        if not model_name in self.model_pipes.keys():
            print('No model with this name exists')
        else:
            if dacc_params is None:
                dacc_params = self.dflt_dacc_params
            dacc = Dacc(
                wf_store=self.wf_store, annot_part=self.annot_part, **dacc_params
            )
            spectra, annots = zip(*dacc.spectr_for_annot_keys(indices))
            spectra = np.array(spectra)
            return run_pipeline(
                spectra, use_ui_map=use_ui_map, **self.model_pipes[model_name]
            )

    def save_run_results(self, results_name, results):
        self.results[results_name] = results
