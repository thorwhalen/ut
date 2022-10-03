"""
It's good practice to write code that can be run as is on someone elses computer (or server).

One thing to avoid is using absolute paths (that refer to your own computer).

One way (but not the only) to use absolute paths, but that are computed on the fly from a relative path, is to use
__file__ to refer to the absolute filepath of the current py file, and then compute the relative absolute path from
there.

The example below show you how you can figure out, in the code itself, what the containing folder of the file containing
 the code you're reading is, and use it to construct an absolute path for a relative reference to a file that's in
 that folder.

"""

import os

current_file_absolute_path = __file__
containing_folder = os.path.dirname(current_file_absolute_path)


print(('\nThe working directory is: {}'.format(os.getcwd())))
print(('\nThe folder where the code of the script is: {}\n'.format(containing_folder)))

relative_path = 'hello_relative_world.txt'
absolute_path_from_relative_path = os.path.join(containing_folder, relative_path)


print(('Checking if filepath {} exists...'.format(absolute_path_from_relative_path)))
assert os.path.isfile(
    absolute_path_from_relative_path
), "Couldn't find file: {}".format(absolute_path_from_relative_path)
print('Yep, it worked')
