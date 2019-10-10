__author__ = 'thorwhalen'


def show(d,nlines=None,cols=None):
    if nlines is None: nlines=len(d)
    if cols is None: cols = d.columns
    if isinstance(cols,str): cols = [cols]
    print(d[:nlines][cols].to_string())


def sw(df, up_rows=10, down_rows=5, left_cols=4, right_cols=3, return_df=False):
    ''' display df data at four corners
        A,B (up_pt)
        C,D (down_pt)
        parameters : up_rows=10, down_rows=5, left_cols=4, right_cols=3
        usage:
            df = pd.DataFrame(np.random.randn(20,10), columns=list('ABCDEFGHIJKLMN')[0:10])
            df.sw(5,2,3,2)
            df1 = df.set_index(['A','B'], drop=True, inplace=False)
            df1.sw(5,2,3,2)
        Note: Found this at http://stackoverflow.com/questions/15006298/how-to-preview-a-part-of-a-large-pandas-dataframe
    '''
    #pd.set_printoptions(max_columns = 80, max_rows = 40)
    ncol, nrow = len(df.columns), len(df)

    # handle columns
    if ncol <= (left_cols + right_cols) :
        up_pt = df.ix[0:up_rows, :]         # screen width can contain all columns
        down_pt = df.ix[-down_rows:, :]
    else:                                   # screen width can not contain all columns
        pt_a = df.ix[0:up_rows,  0:left_cols]
        pt_b = df.ix[0:up_rows,  -right_cols:]
        pt_c = df[-down_rows:].ix[:,0:left_cols]
        pt_d = df[-down_rows:].ix[:,-right_cols:]

        up_pt   = pt_a.join(pt_b, how='inner')
        down_pt = pt_c.join(pt_d, how='inner')
        up_pt.insert(left_cols, '..', '..')
        down_pt.insert(left_cols, '..', '..')

    overlap_qty = len(up_pt) + len(down_pt) - len(df)
    down_pt = down_pt.drop(down_pt.index[list(range(overlap_qty))]) # remove overlap rows

    dt_str_list = down_pt.to_string().split('\n') # transfer down_pt to string list
