__author__ = 'thor'

import numpy as np
import librosa
from ut.sound.util import Sound
from numpy import *
import matplotlib.pyplot as plt


class EigenSound(object):
    def __init__(self, pca_components, mel_kwargs, n_components=None):
        if n_components is not None:
            self.n_components = n_components
            self.pca_components = pca_components[:n_components, :]
        else:
            self.n_components = shape(pca_components)[0]
            self.pca_components = pca_components
        self.mel_kwargs = mel_kwargs

    def melspectr(self, sound):
        if isinstance(sound, Sound):
            sound = (sound.wf, sound.sr)
        if isinstance(sound, tuple):
            # then melspectr is actually the original sound waveform (wf, sr), so get the melspectr
            sound = librosa.feature.melspectrogram(y=sound[0], sr=sound[1], **self.mel_kwargs)
        return sound

    def transform_melspectr(self, sound, display=False):
        eigen_spectr = dot(self.pca_components, log(self.melspectr(sound)))
        if display == True:
            h = plt.matshow(np.flipud(eigen_spectr), cmap='hot_r')
            # h = librosa.display.specshow(data=eigen_spectr, y_axis='mel')
            # plt.colorbar(format='%+02.01f dB')
            return h
        else:
            return eigen_spectr

    def inverse_transform_melspectr(self, eigen_spectr, display=False):
        melspectr = dot(self.pca_components.T, eigen_spectr)
        if display == True:
            h = librosa.display.specshow(data=melspectr, y_axis='mel')
            plt.colorbar(format='%+02.01f dB')
            return h
        else:
            return melspectr

    def plot_two_first_eigens(self, sound, title=True, plot_kwargs={}):
        X = self.transform_melspectr(sound);
        plot_kwargs = dict(plot_kwargs, **{'alpha': 0.3})
        plt.plot(X[0, :], X[1, :], '-o', **plot_kwargs)
        plt.annotate('begining', xy=(X[0, 0], X[1, 0]))
        plt.annotate('end', xy=(X[0, -1], X[1, -1]))
        if title:
            if title is True:
                title = "two first eigensounds"
                if isinstance(sound, Sound):
                    if sound.name:
                        title += "\n of {}".format(sound.name)
            plt.title(title)


class GeneralEigenSound(object):
    def __init__(self, decomp, mel_kwargs):
        self.decomp = decomp
        self.mel_kwargs = mel_kwargs

    def melspectr(self, sound):
        if isinstance(sound, Sound):
            sound = (sound.wf, sound.sr)
        if isinstance(sound, tuple):
            # then melspectr is actually the original sound waveform (wf, sr), so get the melspectr
            sound = librosa.feature.melspectrogram(y=sound[0], sr=sound[1], **self.mel_kwargs)
        return sound

    def transform_melspectr(self, sound, display=False):
        # eigen_spectr = dot(self.decomp.components_, log(self.melspectr(sound)))
        eigen_spectr = self.decomp.transform(log(self.melspectr(sound)).T).T
        if display == True:
            h = plt.matshow(np.flipud(eigen_spectr), cmap='hot_r')
            return h
        else:
            return eigen_spectr

    def inverse_transform_melspectr(self, eigen_spectr, display=False):
        melspectr = dot(self.pca_components.T, eigen_spectr)
        if display == True:
            h = librosa.display.specshow(data=melspectr, y_axis='mel')
            plt.colorbar(format='%+02.01f dB')
            return h
        else:
            return melspectr

    def plot_two_first_eigens(self, sound, title=True, plot_kwargs={}):
        X = self.transform_melspectr(sound);
        plot_kwargs = dict(plot_kwargs, **{'alpha': 0.3})
        plt.plot(X[0, :], X[1, :], '-o', **plot_kwargs)
        plt.annotate('begining', xy=(X[0, 0], X[1, 0]))
        plt.annotate('end', xy=(X[0, -1], X[1, -1]))
        if title:
            if title is True:
                title = "two first eigensounds"
                if isinstance(sound, Sound):
                    if sound.name:
                        title += "\n of {}".format(sound.name)
            plt.title(title)