from itertools import cycle, islice
import re

ascii_alphabet = 'abcdefghijklmnopqrstuvwxyz'
alpha_numerics = 'abcdefghijklmnopqrstuvwxyz0123456789'
vowels = 'aeiou'
consonants = 'bcdfghjklmnpqrstvwxyz'
vowels_and_consonants = (vowels, consonants)


def number_to_multi_base(n, b):
    """
    Convert a number to a multi-base (generalization of base projection).

    Args:
        n: The number to convert
        b: The base to convert it to

    Returns: A list representing the number in the desired base.

    # When b is just one number, it's the base (for e.g. b=2 means binary base)
    >>> number_to_multi_base(3, 2)
    [1, 1]
    >>> number_to_multi_base(4, 2)
    [1, 0, 0]
    >>> number_to_multi_base(5, 2)
    [1, 0, 1]
    # But the cool thing about number_to_multi_base is that you can have a more complex base (any iterable, really)
    >>> number_to_multi_base(11, [2, 3])
    [1, 2, 1]
    >>> number_to_multi_base(12, [2, 3])
    [1, 0, 0, 0]
    >>> number_to_multi_base(13, [2, 3])
    [1, 0, 0, 1]
    >>> number_to_multi_base(14, [2, 3])
    [1, 0, 1, 0]
    >>> number_to_multi_base(15, [2, 3])
    [1, 0, 1, 1]
    >>> number_to_multi_base(16, [2, 3])
    [1, 0, 2, 0]
    """
    if isinstance(b, (int, float)):
        b = [b]
    base = cycle(b)

    if n == 0:
        return [0]
    digits = []
    while n:
        b = next(base)
        digits.append(int(n % b))
        n //= b
    return digits[::-1]


def str_from_num_list(coord, symbols_for_base_idx=vowels_and_consonants, base_phase=0):
    """
    Make a string from the coordinates (a) of a number in a given base system (infered from symbols_for_base_idx and
    base_phase).

    NOTE: symbols_for_base_idx sets should (in most cases) all be disjoint (but this is not validated!)

    Args:
        coord: An array of integers. Coordinates of a number in a given base system
        base_phase: Which base (of symbols_for_base_idx) to start with (and then cycle)
        symbols_for_base_idx: Sets of symbols for each base

    Returns:
        A string (which is the mapping of the number (represented by coord).

    >>> str_from_num_list([1,2,1,2], ['ae', 'xyz'])
    'ezez'
    >>> str_from_num_list([1,2,1,0], ['ae', 'xyz'])
    'ezex'
    >>>
    >>> # [1,2,0,1] is [1,2,1,0], with the last two digits flipped, but you don't get ezxe in the following:
    >>> str_from_num_list([1,2,0,1], ['ae', 'xyz'])
    'ezay'
    """
    n = len(symbols_for_base_idx)
    s = ''
    for letter_idx, collection_idx in zip(coord, islice(cycle(range(n)), base_phase, None)):
        #         print(f"{letter_idx} === {collection_idx}")
        s += symbols_for_base_idx[collection_idx][letter_idx]
    return s


# TODO: Look into coverage. Couldn't produce 'magic' with ['ai', 'mgc'] or ['mgc', 'ai']
def text_for_num(num, symbols_for_base_idx=vowels_and_consonants):
    """
    Map a number to a string.
    The map is bijective (a.k.a. "1-to-1" if the set of symbols in symbols_for_base_idx are non-overlapping.

    Args:
        num: A number to map to text
        symbols_for_base_idx: The sets of symbols to use: A list of strings, each string representing a
            collection of symbols to use in each base.

    Returns:
        A string representing the input number.

    >>> # using the default symbols_for_base_idx (vowels and consonants):
    >>> text_for_num(1060)
    'caca'
    >>> text_for_num(14818)
    'sapu'
    >>> text_for_num(335517)
    'tecon'
    >>>
    >>> # using custom ones:
    >>> text_for_num(153, ['ai', 'gcm'])
    'magic'
    """
    base_cardinalities = list(map(len, symbols_for_base_idx))
    n_bases = len(base_cardinalities)
    base_phase = num % n_bases

    num = (num - base_phase) // n_bases
    base = list(islice(cycle(base_cardinalities), base_phase, n_bases + base_phase))
    coord = number_to_multi_base(num, base)

    return str_from_num_list(coord[::-1], symbols_for_base_idx, base_phase)[::-1]


inf = float('infinity')


def text_to_pronounceable_text(text,
                               symbols_for_base_idx=vowels_and_consonants,
                               captured_alphabet=alpha_numerics,
                               case_sensitive=False,
                               max_word_length=30,
                               artificial_word_sep='_',
                               assert_no_word_sep_in_text=False
                               ):
    """

    Args:
        text: text you want to map
        symbols_for_base_idx: symbols you want to map TO (default is vowels and consonants)
        captured_alphabet: the symbols of the words you want to map FROM (essentially, in contrast to filler characters)
        case_sensitive: Whether the input text should be lower cased before being processed
        max_word_length: The maximum length of a pronounceable word
        artificial_word_sep: The separator to separate pronounceable words when the word is too long
        assert_no_word_sep_in_text: Whether to assert that artificial_word_sep is not already in the input text
            (to avoid clashing and non-invertibility)

    Returns:
        A more pronounceable text, where pronounceable is defined by you, so not my fault if it's not.

    >>> text_to_pronounceable_text('asd8098 098df')
    'izokagamuta osuhoju'
    >>> text_to_pronounceable_text('asd8098 098df', max_word_length=4, artificial_word_sep='_')
    'izo_kaga_muta osu_hoju'
    """
    if not case_sensitive:
        text = text.lower()

    p = re.compile(f'[{captured_alphabet}]+')  # to match the text to be mapped
    anti_p = re.compile(f'[^{captured_alphabet}]+')  # to match the chunks of separator (not matched) text

    matched_text = anti_p.split(text)
    num_of_character = {c: i for i, c in enumerate(captured_alphabet)}  # the numerical mapping of alphabet
    base_n = len(captured_alphabet)
    # function to get the (base_n) number for a chk
    num_of_chk = lambda chk: sum(num_of_character[c] * (base_n ** i) for i, c in enumerate(chk))

    _text_for_num = lambda num: text_for_num(num, symbols_for_base_idx)
    pronounceable_words = [_text_for_num(num_of_chk(chk)) for chk in matched_text]

    if max_word_length < inf:
        def post_process_word(word):
            if len(word) > max_word_length:
                if assert_no_word_sep_in_text:
                    assert artificial_word_sep not in text, \
                        f"Your artificial_word_sep ({artificial_word_sep}) was in the text (so no bijective mapping)"
                r = (len(word) % max_word_length)
                word_suffix = word[:r]
                word_prefix = word[r:]
                word = artificial_word_sep.join(map(''.join, zip(*([iter(word_prefix)] * max_word_length))))
                if word_suffix:
                    word = word_suffix + artificial_word_sep + word
                return word
            else:
                return word

        pronounceable_words = list(map(post_process_word, pronounceable_words))

    separator_text = p.split(text)

    if len(pronounceable_words) < len(separator_text):
        return ''.join(map(''.join, zip(separator_text, pronounceable_words)))
    else:
        return ''.join(map(''.join, zip(pronounceable_words, separator_text)))


class FunTests:
    @staticmethod
    def print_sequences_in_columns(start_num=3000, end_num=3060):
        for i in range(start_num, end_num):
            #     print(f"-----{i}")
            if i % 2:
                print("".join(map(str, (text_for_num(i)))))
            else:
                print("\t" + "".join(map(str, (text_for_num(i)))))


if __name__ == '__main__':
    try:
        import argh
    except ImportError:
        raise ImportError("You don't have argh. You can install it by doing:\n"
                          "     pip install argh\n"
                          "In your terminal/environment,")

    argh.dispatch_command(text_to_pronounceable_text)
