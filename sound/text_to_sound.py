

import os
from numpy import array

import soundfile as sf

from ut.sound.util import Sound

folder_containing_current_file, _ = os.path.split(__file__)
data_folder = os.path.join(folder_containing_current_file, 'data')

DFLT_SR = 44100
DFLT_ALPHABET_STR = "#$%&()*+,-./0123456789:<=>?[\]^_abcdefghijklmnopqrstuvwxyz~"
DFLT_ALPHABET_WAV_FILE = os.path.join(data_folder, 'abcdef_etc.wav')


class TextToSound(object):
    def __init__(self, char_wf, sr=DFLT_SR):
        self.char_wf = char_wf
        self.sr = sr

    @classmethod
    def from_str_and_wav(cls, alphabet_str=DFLT_ALPHABET_STR, alphabet_wav_file=DFLT_ALPHABET_WAV_FILE,
                         wf_dtype='float64'):

        wf, sr = sf.read(alphabet_wav_file, dtype='float64')
        n_chars = len(alphabet_str)
        frm_per_char = len(wf) / n_chars

        char_wf = dict()
        for i, c in enumerate(alphabet_str):
            char_wf[c] = wf[int(i * frm_per_char): int((i + 1) * frm_per_char)]
        char_wf[' '] = array([0] * int(frm_per_char))  # add the space to the character

        return TextToSound(char_wf=char_wf, sr=sr)

    def text_to_wf(self, txt):
        assert set(txt + ' ').issubset(list(self.char_wf.keys())), \
            "Your text needs to only use space and the following characters:\n{}".format(''.join(char_wf))
        wf = []
        for c in txt:
            wf += list(self.char_wf[c])
        return array(wf)

    def text_to_sound(self, txt):
        return Sound(wf=self.text_to_wf(txt), sr=self.sr)
