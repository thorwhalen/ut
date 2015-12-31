from __future__ import division
__author__ = 'thor'


from numpy import *
import os
import re
import librosa
import soundfile as sf
import wave
import contextlib

from IPython.display import Audio
import matplotlib.pyplot as plt
# import numpy as np
from scipy.signal import resample as scipy_signal_resample

# from functools import partial

from ut.util.log import printProgress
from ut.util.pfunc import filter_kwargs_to_func_arguments

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


def is_mono(wf):
    return len(shape(wf)) == 1


def ensure_mono(wf):
    if is_mono(wf):
        return wf
    else:
        return mean(wf, axis=1)
        # return wf[:, 0]


def resample_wf(wf, sr, new_sr):
    return scipy_signal_resample(wf, num=round(len(wf) * new_sr / sr))


def suffix_with_silence(wf, num_silence_pts):
    if is_mono(wf):
        return hstack([wf, zeros(num_silence_pts)])
    else:
        return vstack([wf, zeros((num_silence_pts, 2))])


def prefix_with_silence(wf, num_silence_pts):
    if is_mono(wf):
        return hstack([zeros(num_silence_pts), wf])
    else:
        return vstack([zeros((num_silence_pts, 2)), wf])


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
    wf[random.randint(len(wf))] *= 1.001  # hack to avoid having exactly the same sound twice (creates an Audio bug!!)
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
    plt.plot(linspace(start=0, stop=len(wf)/float(sr), num=len(wf)), wf)


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
    assert len(wf.shape) == 1, "The silence crop is only implemented for mono sounds"
    first_non_zero = argmax(wf != 0)
    last_non_zero = len(wf) - argmin(flipud(wf == 0))
    return wf[first_non_zero:last_non_zero]


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
    must_ensure_mono = kwargs.get('ensure_mono', False)

    if 'offset_s' in kwargs.keys() or 'duration' in kwargs.keys():
        sample_rate = wave.Wave_read(filepath).getframerate()
        start = int(round(kwargs.pop('offset_s', 0) * sample_rate))
        kwargs['start'] = start
        duration = kwargs.pop('duration', None)
        if duration is not None:
            kwargs['stop'] = int(start + round(duration * sample_rate))

    kwargs = filter_kwargs_to_func_arguments(sf.read, kwargs)
    wf, sr = sf.read(filepath, **kwargs)
    if must_ensure_mono:
        wf = ensure_mono(wf)
    return wf, sr

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
        self.wf = wf.copy()
        self.sr = sr
        self.name = name
        self.info = {}

    def copy(self):
        return Sound(wf=self.wf.copy(), sr=self.sr, name=self.name)

    ####################################################################################################################
    # CREATION

    @classmethod
    def from_file(cls, filepath, name=None, **kwargs):
        file_name, extension = os.path.splitext((os.path.basename(filepath)))
        name = name or file_name
        kwargs = dict({'always_2d': False}, **kwargs)

        wf, sr = wf_and_sr_from_filepath(filepath, **kwargs)

        sound = Sound(wf=wf, sr=sr, name=name)

        if extension == '.wav':
            try:
                sound.info = sound_file_info_dict(filepath)
                offset_s = kwargs.get('offset_s', None)
                if offset_s is not None:
                    sound.info['offset_s'] = float(offset_s)
                duration = kwargs.get('duration', None)
                if duration is not None:
                    sound.info['duration'] = float(duration)
                if duration is not None or offset_s is not None:
                    offset_s = offset_s or 0
                    sound.info['frames'] = int((duration - offset_s) * 48000)
                    sound.info.pop('size')
            except Exception:
                pass

        return sound


    @classmethod
    def from_sound_iterator(cls,
                            sound_iterator,
                            name='from_sound_iterator',
                            pre_normalization_function=lambda wf: wf / percentile(abs(wf), 95)):
        """
        Mix all sounds specified in the sound_iterator.

        A sound iterator yields either of these formats:
            * a wave form
            * a Sound object
            * a {sound, offset_s, weight} dict indicating
                offset_s (default 0 seconds): where the sound should be inserted
                weight (default 1): a weight, relative to the other sounds in the iterator, indicating whether the
                "volume" should be increased or decreased before mixing the sound

        Note: All wave forms are normalized before being multiplied by the given weight. The normalization function is
        given by the pre_normalization_function argument (default is no normalization)

        Note: It is assumed that all sounds in the sound_iterator have the same sample rate
        """

        def _mk_sound_mix_spec(sound_mix_spec):
            sound_mix_spec_default = dict(sound=None, offset_s=0, weight=1)
            if isinstance(sound_mix_spec, ndarray):
                sound_mix_spec = dict(sound_mix_spec_default, sound=Sound(wf=sound_mix_spec, sr=None))
            elif hasattr(sound_mix_spec, 'wf'):
                sound_mix_spec = dict(sound_mix_spec_default, sound=sound_mix_spec)
            else:
                sound_mix_spec = dict(sound_mix_spec_default, **sound_mix_spec)
            sound_mix_spec['sound'] = sound_mix_spec['sound'].copy()  # to make sure the user doesn't overwrite it
            sound_mix_spec['sound'].wf = ensure_mono(sound_mix_spec['sound'].wf)
            return sound_mix_spec

        if isinstance(sound_iterator, dict):
            sound_iterator = sound_iterator.values()
        sound_iterator = iter(sound_iterator)
        # compute the weight factor. All input weights will be multiplied by this factor to avoid last sounds having
        # more volume than the previous ones

        # take the first sound as the sound to begin (and accumulate) with. As a result, the sr will be taken from there
        sound_mix_spec = _mk_sound_mix_spec(sound_iterator.next())
        result_sound = sound_mix_spec['sound'].copy()
        result_sound.name = name
        result_sound.info = {}  # we don't want to keep the first sound's info around
        sounds_mixed_so_far = 1
        try:
            while True:
                sound_mix_spec = _mk_sound_mix_spec(sound_iterator.next())
                new_sound_sr = sound_mix_spec['sound'].sr
                assert new_sound_sr is None or new_sound_sr == result_sound.sr, \
                    "All sample rates must be the same to mix sounds: \n" \
                    "   The first sound had sample rate {}, and the {}th one had sample rate{}".format(
                        result_sound.sr, sounds_mixed_so_far + 1, new_sound_sr
                    )
                # divide weight by number of sounds mixed so far, to avoid last sounds having more volume
                # than the previous ones
                weight = sound_mix_spec['weight'] / sounds_mixed_so_far
                # print(weight)
                # offset the new sound
                offset_length = ceil(sound_mix_spec.get('offset_s', 0) * new_sound_sr)
                sound_mix_spec['sound'].wf = prefix_with_silence(sound_mix_spec['sound'].wf, offset_length)

                # finally, mix these sounds
                # print(pre_normalization_function(range(100)))
                result_sound.mix_in(sound_mix_spec['sound'],
                                    weight=weight,
                                    pre_normalization_function=pre_normalization_function)

                # increment the counter
                sounds_mixed_so_far += 1

        except StopIteration:
            pass

        return result_sound

    def save_to_wav(self, filepath=None, samplerate=None, **kwargs):
        samplerate = samplerate or self.sr
        filepath = filepath or (self.name + '.wav')
        sf.write(self.wf, file=filepath, samplerate=samplerate, **kwargs)

    ####################################################################################################################
    # TRANSFORMATIONS

    def ensure_mono(self):
        self.wf = ensure_mono(self.wf)

    def resample(self, new_sr, inplace=False):
        new_wf = resample_wf(self.wf, self.sr, new_sr)
        if not inplace:
            return Sound(wf=new_wf, sr=new_sr, name=self.name)
        else:
            self.wf = new_wf
            self.sr = new_sr

    def crop_with_idx(self, first_idx, last_idx):
        cropped_sound = self.copy()
        cropped_sound.wf = cropped_sound.wf[first_idx:(last_idx + 1)]
        return cropped_sound

    def crop_with_seconds(self, first_second, last_second):
        return self.crop_with_idx(round(first_second * self.sr), round(last_second * self.sr))

    def mix_in(self,
               sound,
               weight=1,
               pre_normalization_function=lambda wf: wf / percentile(abs(wf), 95)):

        # resample sound to match self, if necessary
        if sound.sr != self.sr:
            sound = sound.resample(new_sr=self.sr)
        new_wf = sound.wf.copy()

        # suffix the shortest sound with silence to match lengths
        existing_sound_length = len(self.wf)
        new_sound_length = len(new_wf)
        length_difference = existing_sound_length - new_sound_length
        if length_difference > 0:
            new_wf = suffix_with_silence(new_wf, length_difference)
        elif length_difference < 0:
            self.wf = suffix_with_silence(self.wf, -length_difference)

        # mix the new wf into self.wf
        # print(pre_normalization_function(arange(100)))
        self.wf = weighted_mean([pre_normalization_function(self.wf), 1],
                                [pre_normalization_function(new_wf), weight])

    ####################################################################################################################
    # DISPLAY FUNCTIONS

    def plot_wf(self):
        plot_wf(wf=self.wf.copy(), sr=self.sr, alpha=0.8)

    def hear_sound(self, **kwargs):
        print("{}".format(self.name))
        wf = ensure_mono(self.wf)
        wf[random.randint(len(wf))] *= 1.001  # hack to avoid having exactly the same sound twice (creates an Audio bug)
        return Audio(data=wf, rate=self.sr, **kwargs)

    def display_sound(self, **kwargs):
        self.plot_wf()
        return self.hear_sound(**kwargs)

    def melspectrogram(self, mel_kwargs={}):
        # Let's make and display a mel-scaled power (energy-squared) spectrogram
        # We use a small hop length of 64 here so that the frames line up with the beat tracker example below.
        mel_kwargs = dict(mel_kwargs, **{'n_fft': 2048, 'hop_length': 512, 'n_mels': 128})
        S = librosa.feature.melspectrogram(self.wf, sr=self.sr, **mel_kwargs)
        # Convert to log scale (dB). We'll use the peak power as reference.
        log_S = librosa.logamplitude(S, ref_power=max)
        # Make a new figure
        plt.figure(figsize=(12, 4))
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
