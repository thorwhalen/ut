

__author__ = 'thor'


import pandas as pd
import numpy as np
import itertools
import time
import requests
import argparse

import plotly.plotly as py
import plotly.graph_objs

from requests.exceptions import ConnectionError

def winner_normalizer(raw_probs):
    return raw_probs / raw_probs.max()


# ip: '54.85.63.111:8083'
# account: 'pop10_home@otosense.com', 'h4@otosense.com', 'shower@otosense.com'
# datatypes: 'raw_probabilities, 'devices/status'
def _get_request_url(minutes=5,
                     ip='54.85.63.111:8083',
                     account='generic3@otosense.com',
                     datatype='raw_probabilities'):
    return 'http://{ip}/api/account/{account}/{datatype}/interval/{minutes}'.format(
        ip=ip, account=account, datatype=datatype, minutes=minutes)


def _get_json_data(minutes=5,
                   ip='54.85.63.111:8083',
                   account='generic3@otosense.com',
                   datatype='raw_probabilities',
                   num_of_connection_tries=20):
    request_string = _get_request_url(minutes=minutes, ip=ip, account=account, datatype=datatype)
    for i in range(num_of_connection_tries):
        try:
            response = requests.get(request_string)
            break
        except ConnectionError:
            pass  # just try again

    if response.ok:
        data = response.json().get('data')
        return data


def _get_raw_prob_df_from_json_data(data):
    def extract_from_single_entry(d):
        raw_probs = d.get('data')
        if raw_probs:
            return dict(createdDate=d['createdDate'], **raw_probs)

    data = pd.DataFrame([_f for _f in map(extract_from_single_entry, data) if _f])
    data['createdDate'] = pd.to_datetime(data['createdDate'])
    data = data.sort_values(by='createdDate')
    data = data.set_index('createdDate')
    return data


def get_raw_prob_df(minutes=5, ip='54.85.63.111:8083', account='generic3@otosense.com'):
    data = _get_json_data(minutes=minutes, ip=ip, account=account)
    if data:
        return _get_raw_prob_df_from_json_data(data)


if __name__ == "__main__":
    defaults = dict(
        account="generic3@otosense.com",
        sensitivity=1,
        median_multiplier=2,
        stream_id='h9pmg5ux9z',
        windows_minutes=1,
        debug=0
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--account", type=str,
                        help="account to listen to (default={})".format(defaults['account']),
                        default=defaults['account'])
    parser.add_argument("--sensitivity", type=int,
                        help="all normalized probs will be ^(1/sensitivity) (default={})".format(
                            defaults['sensitivity']),
                        default=defaults['sensitivity'])
    parser.add_argument("--median_multiplier", type=int,
                        help="A number to multiply the median of sound probs by. "
                             "Result will be used as normalizing factor (default={})".format(defaults['median_multiplier']),
                        default=defaults['sensitivity'])
    parser.add_argument("--stream_id", type=str,
                        help="plotly stream_id (default={})".format(defaults['stream_id']),
                        default=defaults['stream_id'])
    parser.add_argument("--windows_minutes", type=int,
                        help="minutes of sliding window (default={})".format(defaults['windows_minutes']),
                        default=defaults['windows_minutes'])
    parser.add_argument("--debug", type=int,
                        help="debug level (default={})".format(defaults['debug']),
                        default=defaults['debug'])

    args = parser.parse_args()
    args = vars(args)

    print(args)

    account = args['account']
    sensitivity = args['sensitivity']
    debug = args['debug']
    stream_id = args['stream_id']
    windows_minutes = args['windows_minutes']
    median_multiplier = args['median_multiplier']

    # Make instance of stream id object
    stream = plotly.graph_objs.Stream(
        token=stream_id,  # (!) link stream id to 'token' key
        maxpoints=80  # (!) keep a max of 80 pts on screen
    )

    mat = np.random.rand(5, 50)

    # Initialize trace of streaming plot by embedding the unique stream_id
    trace1 = plotly.graph_objs.Heatmap(
        z=mat,
        stream=stream  # (!) embed stream id, 1 per trace
    )

    data = plotly.graph_objs.Data([trace1])

    # Add title to layout object
    layout = plotly.graph_objs.Layout(title='Monitoring {}'.format(args['account']))

    # Make a figure object
    fig = plotly.graph_objs.Figure(data=data, layout=layout)

    # (@) Send fig to Plotly, initialize streaming plot, open new tab
    unique_url = py.plot(fig, filename='s7_first-stream', auto_open=False)
    print(("Json request REST url:\n\t{}".format(_get_request_url(minutes=windows_minutes, account=account))))
    print(("Stream image URL:\n\t{}".format(unique_url)))

    # (@) Make instance of the Stream link object,
    # with same stream id as Stream id object
    s = py.Stream(stream_id)

    # (@) Open the stream
    s.open()

    max_sound_probabilities = None

    i = 0
    while True:
        i += 1
        # increment mat
        # mat = hstack([mat[:, 1:], rand(mat.shape[0], 1)])
        data = _get_json_data(minutes=windows_minutes,
                              ip='54.85.63.111:8083',
                              account=account,
                              datatype='raw_probabilities')
        mat = pd.DataFrame([x.get('data') for x in data])

        # if max_sound_probabilities is None:
        #     max_sound_probabilities = mat.median(axis=0) * median_multiplier
        #     # max_sound_probabilities = mat.max(axis=0)
        # else:  # update max_sound_probabilities
        #     max_sound_probabilities = \
        #         pd.DataFrame([max_sound_probabilities, mat.median(axis=0) * median_multiplier]).max()
        #     # max_sound_probabilities = pd.DataFrame([max_sound_probabilities, mat.max(axis=0)]).max()
        #
        # mat /= max_sound_probabilities  # normalize according to max probability

        mat = (mat.T / mat.T.max()).T

        if debug > 0:
            print(i)
            print(max_sound_probabilities)

        sounds = list(mat.columns)
        mat = mat.T.as_matrix()
        mat **= (1 / sensitivity)


        # (@) write to Plotly stream!
        s.write(plotly.graph_objs.Heatmap(z=mat, y=sounds))

        # (!) Write numbers to stream to append current data on plot,
        # write lists to overwrite existing data on plot (more in 7.2).

        time.sleep(0.08)  # (!) plot a point every 80 ms, for smoother plotting

    # (@) Close the stream when done plotting
    s.close()
