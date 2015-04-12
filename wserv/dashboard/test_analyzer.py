__author__ = 'thor'

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analyzer import Analyzer
from params import default

from ut.daf.to import to_html

form_elements = [
    dict(name='your_name', type='text', display="Your Name", value='Unknown'),
    dict(name='max_num', type='number', value=1),
    dict(name='register', type='submit', value='register inputs'),
    dict(name='npts', type='number', value=30, display="num of rand pts"),
    dict(name='graph', type='submit', value="graph it")
]


class TestAnalyzer(Analyzer):
    def __init__(self, input_element_collection=form_elements, work_folder='.'):
        super(TestAnalyzer, self).__init__(input_element_collection)
        self.a = dict()
        self.a['image_html'] = '<img style="box-shadow:         3px 3px 5px 6px #ccc;" src={image_url}>'
        self.work_folder = work_folder

    def register(self, **kwargs):
        self.set_inputs(**kwargs)

    def graph_it(self, **kwargs):
        y = np.random.rand(self.input['npts']) * self.input['max_num']
        fig = plt.figure(figsize=(6, 6))
        plt.plot(y, figure=fig)

        image_name = "TestAnalyzer01.png"
        image_path = os.path.join(self.work_folder, image_name)
        if os.path.exists(image_path):
            os.remove(image_path)
        fig.savefig(image_path, **default['save_fig_params'])
        html = self.a['image_html'].format(image_url=image_path)

        d = pd.DataFrame({'input': self.input.keys(), 'value': self.input.values()})
        html += "<br>\n" + to_html(d, template='box-table-c', index=False, float_format=lambda x: "{:,.0f}".format(x))
        return html

    # def generate_plot_of_traj_and_save_to_file(self, plot_fun=plot, plot_kwargs={}, save_fig_params={}):
    #     fig, ax = plot_fun(**plot_kwargs)
    #     save_fig_params = dict(default['save_fig_params'], **save_fig_params)
    #     fig.savefig(self.a.traj_image_filepath, **save_fig_params)






