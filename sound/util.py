__author__ = 'thor'

import os
import re
import librosa
import soundfile as sf
import wave
import contextlib

from IPython.display import Audio
import matplotlib.pyplot as plt
import numpy as np
# from functools import partial

from ut.util.log import printProgress

wav_text_info_exp = re.compile("^.*WAVEbextZ\x03\x00\x00([^\x00]+)")


def sound_file_info_dict(filepath):
    filename = os.path.basename(filepath)
    (shortname, extension) = os.path.splitext(filename)
    d = {'filepath': filepath,
         'name': shortname,
         'size': os.path.getsize(filepath),
         'ext': extension[1:]
         }
    if extension == '.wav':
        with contextlib.closing(wave.open(filepath, 'r')) as f:
            d['channels'] = f.getnchannels()
            d['sample_width'] = f.getsampwidth()
            d['frames'] = f.getnframes()
            d['frame_rate'] = f.getframerate()
            d['duration'] = d['frames'] / float(d['frame_rate'])
        with file(filepath, 'r') as f:
            text_info = get_wav_text_info(f)
            if text_info is not None:
                d['inner_wav_text'] = text_info

    return d


def ensure_mono(wf):
    if len(np.shape(wf)) == 2:
        return wf[:, 0]
    else:
        return wf


def stereo_to_mono_and_extreme_silence_cropping(source, target, subtype=None, print_progress=False):
    if os.path.isdir(source) and os.path.isdir(target):
        from glob import iglob
        if source[-1] != '/':
            source += '/'
        for i, filepath in enumerate(iglob(source + '*.wav')):
            filename = os.path.basename(filepath)
            if print_progress:
                printProgress("{}: {}".format(i, filename))
            stereo_to_mono_and_extreme_silence_cropping(
                filepath,
                os.path.join(target, filename)
            )
    else:
        wf, sr = wf_and_sr(source)
        wf = ensure_mono(wf)
        wf = crop_head_and_tail_silence(wf)
        sf.write(data=wf, file=target, samplerate=sr, subtype=subtype)


def get_wav_text_info(filespec):
    if isinstance(filespec, basestring):
        with file(filespec, 'r') as fd:
            m = wav_text_info_exp.match(fd.read())
            if m is not None:
                return m.groups()[0]
            else:
                return None
    else:
        m = wav_text_info_exp.match(filespec.read())
        if m is not None:
            return m.groups()[0]
        else:
            return None


def wf_and_sr(*args, **kwargs):
    if len(args) > 0:
        args_0 = args[0]
        if isinstance(args_0, basestring):
            kwargs['filepath'] = args_0
        elif isinstance(args_0, tuple):
            kwargs['wf'], kwargs['sr'] = args_0
    kwargs_keys = kwargs.keys()
    if 'filepath' in kwargs_keys:
        return wf_and_sr_from_filepath(filepath=kwargs['filepath'])
    if 'wf' in kwargs_keys:
        return kwargs['wf'], kwargs['sr']


def hear_sound(*args, **kwargs):
    wf, sr = wf_and_sr(*args, **kwargs)
    try:
        return Audio(data=wf, rate=sr, autoplay=kwargs.get('autoplay', False))
    except ValueError:
        try:
            # just return left audio (stereo PCM signals are unsupported
            return Audio(data=wf[0, :], rate=sr, autoplay=kwargs.get('autoplay', False))
        except:
            return Audio(data=wf[:, 0], rate=sr, autoplay=kwargs.get('autoplay', False))


def plot_wf(*args, **kwargs):
    wf, sr = wf_and_sr(*args, **kwargs)
    plt.plot(np.linspace(start=0, stop=len(wf)/float(sr), num=len(wf)), wf)


def display_sound(*args, **kwargs):
    plot_wf(*args, **kwargs)
    return hear_sound(*args, **kwargs)


def duration_of_wf_and_sr(wf, sr):
    return len(wf) / float(sr)


def n_wf_points_from_duration_and_sr(duration, sr):
    return int(round(duration * sr))


def get_consecutive_zeros_locations(wf, sr, thresh_consecutive_zeros_seconds=0.1):

    thresh_consecutive_zeros = thresh_consecutive_zeros_seconds * sr
    list_of_too_many_zeros_idx_and_len = list()
    cum_of_zeros = 0

    for i in xrange(len(wf)):
        if wf[i] == 0:
            cum_of_zeros += 1  # accumulate
        else:
            if cum_of_zeros > thresh_consecutive_zeros:
                list_of_too_many_zeros_idx_and_len.append({'idx': i - cum_of_zeros, 'len': cum_of_zeros})  # remember
            cum_of_zeros = 0  # reinit
    if cum_of_zeros > thresh_consecutive_zeros:
        list_of_too_many_zeros_idx_and_len.append({'idx': i - cum_of_zeros, 'len': cum_of_zeros})  # remember

    return list_of_too_many_zeros_idx_and_len


def crop_head_and_tail_silence(wf):
    first_non_zero = np.argmax(wf != 0)
    last_non_zero = len(wf) - np.argmin(np.flipud(wf == 0))
    return wf[first_non_zero:last_non_zero]


# def wf_and_sr_of_middle_seconds(filepath, sample_seconds=5.0, pad=False):
    # if sound_seconds is None:
    #     wf, sr = wf_and_sr_from_filepath(filepath)
    #     sound_seconds = duration_of_wf_and_sr(wf, sr)
    # mid_sound = sound_seconds * 0.5
    # if sample_seconds >= sound_seconds:
    #     return wf_and_sr_from_filepath(filepath,
    #                                    offset=mid_sound - sample_seconds * 0.5,
    #                                    duration=sound_seconds)
    # else:
    #     if pad:
    #         if 'sr' not in locals():
    #             wf, sr = wf_and_sr_from_filepath(filepath)
    #         pad_size = n_wf_points_from_duration_and_sr(duration=(sound_seconds - sample_seconds) / 2.0, sr=sr)
    #         wf = np.concatinate((np.zeros(pad_size), wf, np.zeros(pad_size)))
    #     else:
    #         return wf_and_sr_from_filepath(filepath)


def is_wav_file(filepath):
    return os.path.splitext(filepath)[1] == '.wav'


def wav_file_framerate(file_pointer_or_path):
    if isinstance(file_pointer_or_path, basestring):
        file_pointer_or_path = wave.open(file_pointer_or_path)
        frame_rate = file_pointer_or_path.getframerate()
        file_pointer_or_path.close()
    else:
        frame_rate = file_pointer_or_path.getframerate()
    return frame_rate


def wf_and_sr_from_filepath(filepath, **kwargs):
    kwargs = dict({'always_2d': False}, **kwargs)

    if 'offset_s' in kwargs.keys() or 'duration' in kwargs.keys():
        sample_rate = wave.Wave_read(filepath).getframerate()
        start = int(round(kwargs.pop('offset_s', 0) * sample_rate))
        kwargs['start'] = start
        duration = kwargs.pop('duration', None)
        if duration is not None:
            kwargs['stop'] = int(start + round(duration * sample_rate))

    return sf.read(filepath, **kwargs)

    # kwargs = dict({'sr': None}, **kwargs)
    # return librosa.load(filepath, **kwargs)


def wave_form(filepath, **kwargs):
    return wf_and_sr_from_filepath(filepath, **kwargs)[0]


def weighted_mean(yw1, yw2):
    common_len = min([len(yw1[0]), len(yw2[0])])
    a = yw1[0][:common_len]
    b = yw2[0][:common_len]
    return (a * yw1[1] + b * yw2[1]) / (yw1[1] + yw2[1])


def mk_transformed_copies_of_sound_files(source_path_iterator,
                          file_reader=wf_and_sr_from_filepath,
                          transform_fun=None,
                          source_path_to_target_path=None,
                          save_fun=None,
                          onerror_fun=None):
    """
    Gets every filepath
        fed by source_path_iterator one by one,
        reads the file in with file_reader(filepath) to get a wave form and sample rate
        feeds these to the transform_fun(wf, sr), which returns another wf and sr,
        which are passed to save_fun(wf, sr, filepath) to be saved as a sound file,
        the target filepath being computed from source_path through the function source_path_to_target_path(path)
        If there's any errors and a onerror_fun(source_path, e) is given, it will be called instead of raising error
    """
    assert source_path_to_target_path is not None, "You must provide a save_fun (function or target folder)"
    if isinstance(source_path_to_target_path, basestring):
        target_folder = source_path_to_target_path
        assert os.path.exists(target_folder), \
            "The folder {} doesn't exist".format(target_folder)

        def source_path_to_target_path(source_path):
            source_name = os.path.splitext(os.path.basename(source_path))[0]
            return os.path.join(target_folder, source_name + '.wav')

    if save_fun is None:
        def save_fun(wf, sr, filepath):
            sf.write(data=wf, file=filepath, samplerate=sr)

    for source_path in source_path_iterator:
        try:
            wf, sr = file_reader(source_path)
            if transform_fun is not None:
                wf, sr = transform_fun(wf, sr)
            target_path = source_path_to_target_path(source_path)
            save_fun(wf=wf, sr=sr, filepath=target_path)
        except Exception as e:
            if onerror_fun is not None:
                onerror_fun(source_path, e)
            else:
                raise e


class Sound(object):
    def __init__(self, wf, sr, name=''):
        self.wf = wf
        self.sr = sr
        self.name = name

    @classmethod
    def from_file(cls, filepath, name=None, **kwargs):
        name = name or os.path.splitext((os.path.basename(filepath)))[0]
        kwargs = dict({'always_2d': False}, **kwargs)
        y, sr = sf.read(filepath, **kwargs)
        return Sound(wf=y, sr=sr, name=name)

    def plot_wf(self):
        plot_wf(wf=self.wf, sr=self.sr)

    def copy(self):
        return Sound(wf=self.wf.copy(), sr=self.sr, name=self.name)

    def mix_in(self, sound, weight=1):
        self.wf = weighted_mean([self.wf, 1], [sound.y, weight])

    def display_sound(self, **kwargs):
        print("{}".format(self.name))
        self.plot_wf()
        return Audio(data=self.wf, rate=self.sr, **kwargs)

    def melspectrogram(self, mel_kwargs={}):
        # Let's make and display a mel-scaled power (energy-squared) spectrogram
        # We use a small hop length of 64 here so that the frames line up with the beat tracker example below.
        mel_kwargs = dict(mel_kwargs, **{'n_fft': 2048, 'hop_length': 64, 'n_mels': 128})
        S = librosa.feature.melspectrogram(self.wf, sr=self.sr, **mel_kwargs)
        # Convert to log scale (dB). We'll use the peak power as reference.
        log_S = librosa.logamplitude(S, ref_power=np.max)
        # Make a new figure
        plt.figure(figsize=(12,4))
        # Display the spectrogram on a mel scale
        # sample rate and hop length parameters are used to render the time axis
        librosa.display.specshow(log_S, sr=self.sr, hop_length=mel_kwargs['hop_length'],
                                 x_axis='time', y_axis='mel')
        # Put a descriptive title on the plot
        plt.title('mel power spectrogram of "%s"' % self.name)
        # draw a color bar
        plt.colorbar(format='%+02.0f dB')
        # Make the figure layout compact
        plt.tight_layout()