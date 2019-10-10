__author__ = 'thor'


def drop_duplicates_pipe(val_cols, gr_cols=None, take_last=False, no_id=True):
    # input processing
    val_cols, gr_cols = get_val_and_gr_cols(val_cols, gr_cols, no_id)
    if take_last:
        first_or_last = '$last'
    else:
        first_or_last = '$first'
    # making the util dicts
    group_dict = dict(list({'_id': x_to_x_dict(gr_cols)}.items()) +
                      list({k: {first_or_last: '$' + k} for k in val_cols}.items()))
    project_dict = dict(list({'_id': 0}.items()) +
                        list({k: '$_id.' + k for k in gr_cols}.items()) +
                        list(x_to_x_dict(val_cols).items()))
    return [{'$group': group_dict}, {'$project': project_dict}]


def popped_unwind_pipe(unwind_field, unwind_sub_fields, other_fields=None, no_id=True):
    if other_fields is None and not isinstance(unwind_sub_fields, list):
        other_fields = unwind_sub_fields
    unwind_sub_fields = get_subfields(c=unwind_sub_fields, field=unwind_field)
    other_fields = list(set(get_fields(other_fields, no_id=no_id)).difference([unwind_field]))
    unwind_projection = {k: '${unwind_field}.{unwind_subfield}'.format(
                unwind_field=unwind_field, unwind_subfield=k) for k in unwind_sub_fields}
    other_projection = {k: 1 for k in other_fields}
    return [{'$unwind': '$' + unwind_field},
            {'$project': dict(list(unwind_projection.items()) + list(other_projection.items()))}]


###### UTILS #######################################################

def x_to_x_dict(x):
    return {k: '$' + k for k in x}


def get_val_and_gr_cols(val_cols, gr_cols, no_id=True):
    cols = get_fields(val_cols, no_id=no_id)
    gr_cols = gr_cols or cols
    val_cols = list(set(cols).difference(gr_cols))
    return val_cols, gr_cols


def get_fields(c, no_id=True):
    if c is None:
        return list()
    else:
        if isinstance(c, list):
            if isinstance(c[0], str):
                fields = c
            else:
                fields = get_fields(c[0])
        elif isinstance(c, dict):
            fields = list(c.keys())
        else:
            try:  # assume it's a collection
                fields = list(c.find_one().keys())
            except AttributeError:  # assume it's a cursor
                fields = list(c.collection.find_one().keys())
        if no_id:
            return list(set(fields).difference(['_id']))
        else:
            return fields


def get_subfields(c, field=None, no_id=True):
    if isinstance(c, list) and isinstance(c[0], str):
        if no_id:
            return list(set(c).difference(['_id']))
        else:
            return c
    else:
        # find the actual list of fields c in c
        if isinstance(c, list) and isinstance(c[0], dict):
            c = c[0]
        elif isinstance(c, dict):
            if field:
                try:
                    c = list(c[field].keys())
                except AttributeError:
                    c = list(c[field][0].keys())
            else:
                c = list(c.keys())
        else:
            try:  # assume it's a collection
                c = c.find_one()
            except AttributeError:  # assume it's a cursor
                c = c.collection.find_one()
        # now call get_subfields on THAT c:
        return get_subfields(c, field=field, no_id=no_id)







