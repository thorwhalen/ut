__author__ = 'thor'

from numpy import *
import numpy as np
import librosa
from sklearn.base import BaseEstimator, TransformerMixin

from ut.sound.util import resample_wf
from ut.sound.util import Sound
import matplotlib.pyplot as plt

from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class DeAmplitudedEigenSound(BaseEstimator, TransformerMixin):
    """
    Transforms sound waves to eigen decomposition of the log of the melspectrogram minus the log of the amplitude sums
    (the columns of the melspectrogram) enhanced with the differential of the log of the amplitude array.

    The idea is to normalize the melspectrogram columns by the amplitude, so to reduce non-meaningful spectrogram
    variation, but keeping the sequence of amplitude around (because in itself, it carries a lot of meaning).

    An (incremental) decomposition of the matrix formed from the normalized log10 melspectrogram will be fit,
    and the fit will also learn the mean and variance of the diff of the amplitude sequence (which also carries meaning)
    """

    def __init__(
        self,
        spectr_decomp=IncrementalPCA(),
        amp_diff_scaler=StandardScaler(),
        mel_kwargs={'n_fft': 2048, 'hop_length': 512, 'n_mels': 128},
        common_sr=44100,
        min_amp=1e-10,
    ):
        self.spectr_decomp = spectr_decomp
        self.amp_diff_scaler = amp_diff_scaler
        self.mel_kwargs = mel_kwargs
        self.common_sr = common_sr
        self.min_amp = min_amp
        self.log10_min_amp = log10(min_amp)

    def melspectrogram(self, wf, sr):
        if sr != self.common_sr:
            wf = resample_wf(wf, sr, new_sr=self.common_sr)
        return librosa.feature.melspectrogram(wf, sr=self.common_sr, **self.mel_kwargs)

    def deamplituded_mel(self, mel_spectrogram):
        mel_spectrogram = np.maximum(mel_spectrogram, self.min_amp)
        amplitude_seq = log10(np.sum(mel_spectrogram, axis=0))
        return log10(mel_spectrogram) - amplitude_seq, diff(amplitude_seq)

    def partial_fit(self, X, y=None):
        _deamplituded_mel, diff_amplitude_seq = self.deamplituded_mel(X)
        self.spectr_decomp.partial_fit(_deamplituded_mel.T)
        self.amp_diff_scaler.partial_fit(diff_amplitude_seq.reshape(-1, 1))
        return self

    def wf_partial_fit(self, wf, sr):
        return self.partial_fit(self.melspectrogram(wf, sr))

    def transform(self, X):
        _deamplituded_mel, diff_amplitude_seq = self.deamplituded_mel(X)
        return np.hstack(
            (
                self.spectr_decomp.transform(_deamplituded_mel.T),
                self.amp_diff_scaler.transform(
                    np.vstack((self.log10_min_amp, diff_amplitude_seq.reshape(-1, 1)))
                ),
            )
        )

    def inverse_transform(self, X):
        eigen_mat = X[:, :-1]
        mel_spectrogram = self.spectr_decomp.inverse_transform(eigen_mat)

        amp_diff = self.amp_diff_scaler.inverse_transform(X[:, -1])
        amplitude_seq = cumsum(amp_diff)

        return (mel_spectrogram.T + amplitude_seq.T).T

    def transform_as_dict(self, X):
        _deamplituded_mel, diff_amplitude_seq = self.deamplituded_mel(X)
        return {
            'scaled_amp_diff': self.amp_diff_scaler.transform(
                np.vstack((self.log10_min_amp, diff_amplitude_seq.reshape(-1, 1)))
            ),
            'eigen': self.spectr_decomp.transform(_deamplituded_mel.T),
        }

    def wf_transform(self, wf, sr):
        return self.transform(self.melspectrogram(wf, sr))

    def wf_transform_as_dict(self, wf, sr):
        return self.transform_as_dict(self.melspectrogram(wf, sr))

    def plot_transform_for_wf(self, wf, sr, **kwargs):
        kwargs.setdefault('aspect', 'auto')
        kwargs.setdefault('origin', 'lower')
        kwargs.setdefault('interpolation', 'nearest')

        fp = self.wf_transform_as_dict(wf, sr)
        eigens = fp['eigen'].T
        kwargs.setdefault('cmap', plt.get_cmap('coolwarm'))

        fig, ax1 = plt.subplots()

        plt.imshow(eigens, axes=ax1, **kwargs)
        plt.yticks([])

        ax2 = ax1.twinx()
        ax2.plot(cumsum(fp['scaled_amp_diff']), 'k-')
        plt.yticks([])
        plt.axis('tight')


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
            sound = librosa.feature.melspectrogram(
                y=sound[0], sr=sound[1], **self.mel_kwargs
            )
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
        X = self.transform_melspectr(sound)
        plot_kwargs = dict(plot_kwargs, **{'alpha': 0.3})
        plt.plot(X[0, :], X[1, :], '-o', **plot_kwargs)
        plt.annotate('begining', xy=(X[0, 0], X[1, 0]))
        plt.annotate('end', xy=(X[0, -1], X[1, -1]))
        if title:
            if title is True:
                title = 'two first eigensounds'
                if isinstance(sound, Sound):
                    if sound.name:
                        title += '\n of {}'.format(sound.name)
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
            sound = librosa.feature.melspectrogram(
                y=sound[0], sr=sound[1], **self.mel_kwargs
            )
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
        X = self.transform_melspectr(sound)
        plot_kwargs = dict(plot_kwargs, **{'alpha': 0.3})
        plt.plot(X[0, :], X[1, :], '-o', **plot_kwargs)
        plt.annotate('begining', xy=(X[0, 0], X[1, 0]))
        plt.annotate('end', xy=(X[0, -1], X[1, -1]))
        if title:
            if title is True:
                title = 'two first eigensounds'
                if isinstance(sound, Sound):
                    if sound.name:
                        title += '\n of {}'.format(sound.name)
            plt.title(title)
