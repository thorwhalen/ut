from __future__ import division
__author__ = 'thor'


from numpy import hstack, vstack, zeros, shape, mean, random, linspace
from numpy import argmax, argmin, flipud, percentile, ndarray, ceil
import numpy as np
import os
import re
# import librosa
from librosa import load as librosa_load
from librosa.feature import melspectrogram as librosa_feature_melspectrogram
from librosa import logamplitude as librosa_logamplitude
from librosa.display import specshow as librosa_display_specshow
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


def complete_sref(sref):
    """
    Complete sref dict with missing fields, if any
    """
    sref = dict({'offset_s': 0.0}, **sref)  # takes care of making a copy (so doesn't overwrite sref)
    if 'duration' not in sref.keys():
        if is_wav_file(sref['filepath']):
            sref['duration'] = get_duration_of_wav_file(sref['filepath'])
        else:
            sound = Sound.from_file(sref['filepath'])
            sref['duration'] = duration_of_wf_and_sr(sound.wf, sound.sr) - sref['offset_s']
    return sref


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


def get_frame_rate_of_wav_file(filepath):
    with contextlib.closing(wave.open(filepath, 'r')) as f:
        return f.getframerate()


def get_duration_of_wav_file(filepath):
    with contextlib.closing(wave.open(filepath, 'r')) as f:
        return f.getnframes() / f.getframerate()


def is_mono(wf):
    return len(shape(wf)) == 1


def ensure_mono(wf):
    if is_mono(wf):
        return wf
    else:
        return mean(wf, axis=1)
        # return wf[:, 0]


def resample_wf(wf, sr, new_sr):
    # return round(len(wf) * new_sr / sr)
    return scipy_signal_resample(wf, num=int(round(len(wf) * new_sr / sr)))


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

    must_ensure_mono = kwargs.pop('ensure_mono', True)

    if is_wav_file(filepath):
        kwargs = dict({'always_2d': False}, **kwargs)
        if 'offset_s' in kwargs.keys() or 'duration' in kwargs.keys():
            sample_rate = wave.Wave_read(filepath).getframerate()
            start = int(round(kwargs.pop('offset_s', 0) * sample_rate))
            kwargs['start'] = start
            duration = kwargs.pop('duration', None)
            if duration is not None:
                kwargs['stop'] = int(start + round(duration * sample_rate))

        kwargs = filter_kwargs_to_func_arguments(sf.read, kwargs)
        wf, sr = sf.read(filepath, **kwargs)
    else:
        kwargs['offset'] = kwargs.pop('offset_s', 0.0)
        wf, sr = librosa_load(filepath, **kwargs)

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


def plot_melspectrogram(spect_mat, sr=44100, hop_length=512, name=None):
    # Make a new figure
    plt.figure(figsize=(12, 4))
    # Display the spectrogram on a mel scale
    # sample rate and hop length parameters are used to render the time axis
    librosa_display_specshow(spect_mat, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
    # Put a descriptive title on the plot
    if name is not None:
        plt.title('mel power spectrogram of "{}"'.format(name))
    else:
        plt.title('mel power spectrogram')
    # draw a color bar
    plt.colorbar(format='%+02.0f dB')
    # Make the figure layout compact
    plt.tight_layout()


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
        # kwargs = dict({'always_2d': False, 'ensure_mono': True}, **kwargs)

        wf, sr = wf_and_sr_from_filepath(filepath, **kwargs)

        if name is None:
            name = filepath

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
    def from_sref(cls, sref):
        wf, sr = wf_and_sr_from_filepath(**complete_sref(sref))
        # filepath = sref['filepath']
        # sample_rate = wave.Wave_read(sref['filepath']).getframerate()
        # start = int(round(sref.get('offset_s', 0) * sample_rate))
        # duration = sref.get('duration', None)
        # kwargs = {}
        # if duration is not None:
        #     kwargs = {'stop': int(start + round(duration * sample_rate))}
        # wf, sr = sf.read(filepath, always_2d=False, start=start, **kwargs)
        return Sound(wf, sr, name=sref.get('name', sref['filepath']))

    @classmethod
    def from_sound_mix_spec(cls,
                            sound_mix_spec,
                            name='from_sound_mix_spec',
                            pre_normalization_function=lambda wf: wf / percentile(abs(wf), 95)):
        """
        Mix all sounds specified in the sound_mix_spec.

        A sound_mix_spec is an iterator that yields either of these formats:
            * a wave form
            * a Sound object
            * (This is the complete specification) a {sound, offset_s, weight} dict indicating
                offset_s (default 0 seconds): where the sound should be inserted
                weight (default 1): a weight, relative to the other sounds in the iterator, indicating whether the
                "volume" should be increased or decreased before mixing the sound

        Note: All wave forms are normalized before being multiplied by the given weight. The normalization function is
        given by the pre_normalization_function argument (default is no normalization)

        Note: If some of the sounds in the sound_mix_spec have different sample rates, they will be resampled to the
        sample rate of the first sound encountered. This process requires (not so fast) fast fourrier transform,
        so better have the same sample rate.
        """

        def _mk_sound_mix_spec(_sound_mix_spec):
            sound_mix_spec_default = dict(sound=None, offset_s=0, weight=1)
            _sound_mix_spec = _sound_mix_spec.copy()
            if isinstance(_sound_mix_spec, dict):  # if sound_mix_spec is a dict...
                if 'filepath' in _sound_mix_spec.keys():  # ... and it has a 'filepath' key...
                    sref = _sound_mix_spec  # ... assume it's an sref...
                    sound = Sound.from_sref(sref) # ... and get the sound from it, and make an actual sound_mix_spec
                    _sound_mix_spec = dict(sound_mix_spec_default, sound=sound)
                else:  # If it's not an sref...
                    sound = _sound_mix_spec['sound']  # ... assume it has a sound key
                    if isinstance(sound, dict):  # and if that "sound" is an sref, replace it by a actual sound object
                        _sound_mix_spec['sound'] = Sound.from_sref(_sound_mix_spec['sound'])
                        _sound_mix_spec = dict(sound_mix_spec_default, **_sound_mix_spec)
            elif isinstance(_sound_mix_spec, ndarray):
                _sound_mix_spec = dict(sound_mix_spec_default, sound=Sound(wf=_sound_mix_spec, sr=None))
            elif hasattr(_sound_mix_spec, 'wf'):
                _sound_mix_spec = dict(sound_mix_spec_default, sound=_sound_mix_spec)
            else:
                _sound_mix_spec = dict(sound_mix_spec_default, **_sound_mix_spec)
            _sound_mix_spec['sound'] = _sound_mix_spec['sound'].copy()  # to make sure the we don't overwrite it in manip
            _sound_mix_spec['sound'].wf = ensure_mono(_sound_mix_spec['sound'].wf)
            # print(sound_mix_spec)
            return _sound_mix_spec

        # if the sound_iterator is a dict, take its values (ignore the keys)
        if isinstance(sound_mix_spec, dict):
            sound_mix_spec = sound_mix_spec.values()
        sound_mix_spec = iter(sound_mix_spec)
        # compute the weight factor. All input weights will be multiplied by this factor to avoid last sounds having
        # more volume than the previous ones

        # take the first sound as the sound to begin (and accumulate) with. As a result, the sr will be taken from there
        spec = _mk_sound_mix_spec(sound_mix_spec.next())
        result_sound = spec['sound']
        result_sound_sr = result_sound.sr  # will be the final sr, and all other sounds will be resampled to it
        result_sound.name = name
        result_sound.info = {}  # we don't want to keep the first sound's info around
        # offset the sound by required amount
        offset_length = ceil(spec.get('offset_s', 0) * result_sound_sr)
        result_sound.wf = prefix_with_silence(result_sound.wf, offset_length)

        # all subsequent weights should be multiplied by a weight_factor,
        # since the accumulating sound is considered to be of unit weight in the Sound.mix_in() method:
        weight_factor = 1 / spec.get('weight', 1.0)
        # initialize sound counter
        sounds_mixed_so_far = 1
        try:
            while True:
                spec = _mk_sound_mix_spec(sound_mix_spec.next())

                # resample sound to match self, if necessary
                #    (mix_in() method does it, but better do it before,
                #       because we're probably going to prefix this sound with silence, so it'll be longer)
                if spec['sound'].sr != result_sound.sr:
                    spec['sound'] = spec['sound'].resample(new_sr=result_sound.sr)

                # divide weight by number of sounds mixed so far, to avoid last sounds having more volume
                # than the previous ones
                weight = weight_factor * spec['weight'] / sounds_mixed_so_far
                # print(weight)
                # offset the new sound
                # print sound_mix_spec['sound_tag'], sound_mix_spec.get('offset_s')
                offset_length = ceil(spec.get('offset_s', 0) * result_sound_sr)
                spec['sound'].wf = prefix_with_silence(spec['sound'].wf, offset_length)

                # finally, mix these sounds
                result_sound.mix_in(spec['sound'],
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

    def display(self, **kwargs):
        self.melspectrogram()
        return self.hear_sound(**kwargs)

    def melspectrogram(self, mel_kwargs={}, plot_it=True):
        # Let's make and display a mel-scaled power (energy-squared) spectrogram
        # We use a small hop length of 64 here so that the frames line up with the beat tracker example below.
        mel_kwargs = dict(mel_kwargs, **{'n_fft': 2048, 'hop_length': 512, 'n_mels': 128})
        S = librosa_feature_melspectrogram(self.wf, sr=self.sr, **mel_kwargs)
        # Convert to log scale (dB). We'll use the peak power as reference.
        log_S = librosa_logamplitude(S, ref_power=np.max)
        if plot_it:
            plot_melspectrogram(log_S, sr=self.sr, hop_length=mel_kwargs['hop_length'], name=self.name)
        return log_S

    def specshow(self, data=None, hop_length=512, x_axis=None, y_axis=None,
                 n_xticks=5, n_yticks=5, fmin=None, fmax=None, bins_per_octave=12,
                 tmin=16, tmax=240, freq_fmt='Hz', time_fmt=None, **kwargs):

        if data is None:
            data = self.melspectrogram(plot_it=False)

        return librosa_display_specshow(
            data, sr=self.sr, hop_length=hop_length, x_axis=x_axis, y_axis=y_axis,
            n_xticks=n_xticks, n_yticks=n_yticks, fmin=fmin, fmax=fmax, bins_per_octave=bins_per_octave,
            tmin=tmin, tmax=tmax, freq_fmt=freq_fmt, time_fmt=time_fmt, **kwargs
        )

    ####################################################################################################################
    # MISC
    def duration(self):
        return duration_of_wf_and_sr(self.wf, self.sr)

    def wf_sr_dict(self):
        return {'wf': self.wf, 'sr': self.sr}

    def wf_sr_tuple(self):
        return self.wf, self.sr