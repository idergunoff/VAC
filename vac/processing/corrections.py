"""Manual curve operations. Bounds are inclusive for curves, exclusive for class intervals."""
from copy import deepcopy
from scipy.interpolate import interp1d
from ..calculation import get_nearest_value

def corr_sig(source, x1, x2, y, shift, linear):
    table = source.copy(deep=True)
    x1_index = table.index[
        table['depth'] == get_nearest_value(table['depth'], x1)].tolist()[0]
    x2_index = table.index[
        table['depth'] == get_nearest_value(table['depth'], x2)].tolist()[0]
    if shift:
        coef_shift = table['first_sig'].iloc[x1_index:x2_index+1].max() - y
        table['corr_sig'] = table['first_sig'].copy()
        table['corr_sig'].iloc[x1_index:x2_index+1] = table['first_sig'].iloc[x1_index:x2_index+1] - coef_shift
    else:
        table['plus_sig'] = table['first_sig'] + abs(table['first_sig'].min()) + 1
        corr_coeff_max = (y + abs(table['first_sig'].min()) + 1)/\
                     table['plus_sig'].iloc[x1_index:x2_index+1].max()
        xmax_index = table.index[table['plus_sig'] == get_nearest_value(table['plus_sig'],
                                                                                  table['plus_sig'].iloc[
                                                                                  x1_index:x2_index + 1].max())].tolist()[0]
        if linear:
            interp_method = 'linear'
        else:
            interp_method = 'quadratic'
        if x1_index == xmax_index:
            corr_coeff = interp1d([x1_index, x2_index], [corr_coeff_max, 1], kind='linear')
        elif xmax_index == x2_index:
            corr_coeff = interp1d([x1_index, x2_index], [1, corr_coeff_max], kind='linear')
        else:
            corr_coeff = interp1d([x1_index, xmax_index, x2_index], [1, corr_coeff_max, 1], kind=interp_method)
        table['corr_coeff'].iloc[x1_index:x2_index+1] = corr_coeff(range(x1_index, x2_index+1))
        table['plus_sig'] = table['plus_sig']*table['corr_coeff']
        table['corr_sig'] = table['plus_sig'] - abs(table['first_sig'].min()) - 1
    return table


def corr_sig_bottom(source, x1, x2, y, shift, linear):
    table = source.copy(deep=True)
    x1_index = table.index[
        table['depth'] == get_nearest_value(table['depth'], x1)].tolist()[0]
    x2_index = table.index[
        table['depth'] == get_nearest_value(table['depth'], x2)].tolist()[0]
    if shift:
        coef_shift = table['first_sig'].iloc[x1_index:x2_index+1].min() - y
        table['corr_sig'] = table['first_sig'].copy()
        table['corr_sig'].iloc[x1_index:x2_index+1] = table['first_sig'].iloc[x1_index:x2_index+1] - coef_shift
    else:
        table['plus_sig'] = table['first_sig'] + abs(table['first_sig'].min()) + 1
        corr_coeff_max = (y + abs(table['first_sig'].min()) + 1) / \
                         table['plus_sig'].iloc[x1_index:x2_index+1].min()
        xmax_index = table.index[table['plus_sig'] == get_nearest_value(table['plus_sig'],
                                                                                  table['plus_sig'].iloc[
                                                                                  x1_index:x2_index+1].min())].tolist()[0]
        if linear:
            interp_method = 'linear'
        else:
            interp_method = 'quadratic'
        if x1_index == xmax_index:
            corr_coeff = interp1d([x1_index, x2_index], [corr_coeff_max, 1], kind='linear')
        elif xmax_index == x2_index:
            corr_coeff = interp1d([x1_index, x2_index], [1, corr_coeff_max], kind='linear')
        else:
            corr_coeff = interp1d([x1_index, xmax_index, x2_index], [1, corr_coeff_max, 1], kind=interp_method)
        table['corr_coeff'].iloc[x1_index:x2_index+1] = corr_coeff(range(x1_index, x2_index+1))
        table['plus_sig'] = table['plus_sig'] * table['corr_coeff']
        table['corr_sig'] = table['plus_sig'] - abs(table['first_sig'].min()) - 1
    return table


def fix_corr(source):
    table = source.copy(deep=True)
    table['first_sig'] = table['corr_sig'].copy()
    table['corr_coeff'] = 1
    return table


def plus_func(source):
    table = source.copy(deep=True)
    table['corr_sig'] = table['first_sig'] + table['func']
    return table


def add_interval(intervals, depth, minimum, maximum, kind):
    updated = deepcopy(intervals)
    first = depth.index[depth == get_nearest_value(depth, minimum)].tolist()[0]
    last = depth.index[depth == get_nearest_value(depth, maximum)].tolist()[0]
    getattr(updated, kind).append([first, last])
    return updated
