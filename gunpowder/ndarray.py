import numpy as np

def replace(array, old_values, new_values):
    '''Replace all occurences of ``old_values[i]`` with ``new_values[i]`` in the
    given array.'''

    old_values = np.array(old_values)
    new_values = np.array(new_values)

    values_map = np.arange(int(array.max() + 1), dtype=new_values.dtype)
    values_map[old_values] = new_values

    return values_map[array]
