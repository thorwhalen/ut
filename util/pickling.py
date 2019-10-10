

import sys
import pickle


def load_pickle_that_has_been_saved_with_a_different_module(
    module_string_of_current_pickle, module_to_load_under, pickle_file_path,
    save_under_new_module=False):
    # map the old ref to the new module (module_to_load_under must be imported!)
    sys.modules[module_string_of_current_pickle] = module_to_load_under
    obj = pickle.load(open(pickle_file_path, 'r'))  # and it works!
    if save_under_new_module:
        # save it under the new module name now!
        pickle.dump(obj, open(pickle_file_path, 'w'))
    return obj