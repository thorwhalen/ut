

__author__ = 'thor'

from numpy import *

from .util import ensure_mono


def _get_wf_and_sr_from_sound(sound):
    """
    Get wf (mono wave form) and sr (sample rate) from sound specification.
    sound could be
        a waveform wf,
        a tuple (wf, sr) where sr is the sample rate, or
        any object that has .wf and .sr attributes.
    """
    if isinstance(sound, tuple) and len(sound) == 2:
        wf, sr = sound
    elif isinstance(sound, ndarray):
        wf = sound
        sr = None
    elif hasattr(sound, 'wf'):
        wf = sound.wf
        sr = None if not hasattr(sound, 'sr') else sound.sr
    else:
        raise TypeError("Unknown sound type")

    return ensure_mono(wf), sr


def silence_interval_indices(sound,
                             min_length_of_silence_interval,
                             max_abs_wf_threshold_func_for_silence=lambda wf: 0.01 * percentile(wf, 99)):
    """
    Returns a list of pairs indicating where "long" intervals of silence happen in the sound.

    What is "sound"? See _get_wf_and_sr_from_sound function. It could be a waveform wf, a tuple (wf, sr)
        where sr is the sample rate, or any object that has .wf and .sr attributes.
        Note that:
            * if sr is not given, we will deal with (wf) integer indices. That is, min_length_of_silence_interval will
            be expressed as array lengths, and the pairs returned by the function will contain indices into wf
            * if sr is given, the index space will be "continuous" time. That is, min_length_of_silence_interval will be
            expressed in seconds, and the function will return intervals expressed as time intervals.

    What is "long"? min_length_of_silence_interval tells you that (either as an array length, or duration)

    What is silence? Good question. They're two problems here:
        (1) Should only null values of the wave form be considered as silence, or should any relatively and
            significantly small values be considered silence?
        (2) Small (even null) absolute wf values happen naturally, since wave forms wiggle between negative and
            positive values. But these small values are transitory, so don't appear in bursts, except that...
        (3) Often silence can be part of, and even defining, the sound itself. One beep of an alarm is just a beep. It's
            the beep-pause-beep pattern that defines the (or more precisely, that type of) alarm.
        That's why we provide the min_length_of_silence_interval: So that you can play around with the definition of
         silence according to your needs. Regarding (1), we also provide you with the
         max_abs_wf_threshold_func_for_silence argument, which is a function to be applied to abs(wf) to get a threshold
         of abs(wf) under which the point will be considered as a silent point.
         The wave form values that will be candidates for inclusion in silence intervals are those whose absolute values
         are smaller than max_abs_wf_threshold_func_for_silence(wf).
    """
    wf, sr = _get_wf_and_sr_from_sound(sound)

    if callable(max_abs_wf_threshold_func_for_silence):
        max_abs_wf_for_silence = max_abs_wf_threshold_func_for_silence(abs(wf))
    else:
        max_abs_wf_for_silence = max_abs_wf_threshold_func_for_silence

    if sr is not None:
        min_length_of_silence_interval = round(sr * min_length_of_silence_interval)
    silence_bitmap = abs(wf) <= max_abs_wf_for_silence

    silence_intervals = list()
    # cumulating_silence is a boolean state indicating whether we're accumulating a silence period...
    cumulating_silence = silence_bitmap[0]  # ... initialize it to the first point of the wave form
    if cumulating_silence == True:
        silence_cumul = 1
        interval_start = 0
    else:
        silence_cumul = 1

    for i, is_silent in enumerate(silence_bitmap[1:], 1):
        if cumulating_silence == True:  # we're in a silence state
            if is_silent:  # just increment the silence count
                silence_cumul += 1
            else:  # Switch to "noise state": a period of silence just ended...
                if silence_cumul > min_length_of_silence_interval:  # ... if it's big enough
                    silence_intervals.append((interval_start, i - 1))
                # else forget this interval
                cumulating_silence = False  # switch to the "not cumulating silence" state
                silence_cumul = 0  # reset the silence cumulator
        else:  # we're in a noise state
            if is_silent:  # Switch to "silence state"
                cumulating_silence = True
                interval_start = i
                silence_cumul = 1

                # if cumulating_silence is False, do nothing. Just wait for some silence.

    # wrap up (make sure we get silence periods that end the wave form)
    if is_silent and cumulating_silence:
        if silence_cumul > min_length_of_silence_interval:  # ... if it's big enough
            silence_intervals.append((interval_start, i - 1))

    if sr is None:
        return silence_intervals
    else:
        return [(i / sr, j / sr) for i, j in silence_intervals]


def _non_silence_interval_indices_from_silence_intervals(array_length, silence_intervals, min_interval_length=2):
    if len(silence_intervals) == 0:
        return [(0, array_length - 1)]
    else:
        # adding fake silence intervals in the beggining of silence_intervals
        if silence_intervals[0][0] != 0:
            silence_intervals = [(nan, -1)] + silence_intervals
        if silence_intervals[-1][1] == array_length - 1:
            array_length = silence_intervals[-1][0]
            silence_intervals = silence_intervals[:-1]

        # adding fake silence intervals at the end of silence_intervals
        silence_intervals = silence_intervals + [(array_length, nan)]

        t = list(zip(*silence_intervals))
        return [(i + 1, j - 1) for i, j in zip(t[1][:-1], t[0][1:]) if (j - i - 1) >= min_interval_length]


def non_silence_interval_indices(sound,
                                 min_length_of_silence_interval,
                                 max_abs_wf_threshold_func_for_silence=lambda wf: 0.01 * percentile(wf, 99),
                                 min_interval_length=None):
    """
    Returns the interval borders of the wave form that correspond to sound not interrupted by periods of significant
    silence. How long these omitted silence periods have to be and what is silence? See the doc for the function:
        silence_interval_indices
    """

    wf, sr = _get_wf_and_sr_from_sound(sound)
    min_interval_length = min_interval_length or 2
    if sr is not None:
        min_length_of_silence_interval = round(sr * min_length_of_silence_interval)
        min_interval_length = min_interval_length or 0.01
        min_interval_length = sr * min_interval_length
    else:
        min_interval_length = min_interval_length or 2

    non_silence_intervals = _non_silence_interval_indices_from_silence_intervals(
        array_length=len(wf),
        silence_intervals=silence_interval_indices(wf, min_length_of_silence_interval,
                                                   max_abs_wf_threshold_func_for_silence),
        min_interval_length=min_interval_length
    )

    if sr is None:
        return non_silence_intervals
    else:
        return [(i / sr, j / sr) for i, j in non_silence_intervals]