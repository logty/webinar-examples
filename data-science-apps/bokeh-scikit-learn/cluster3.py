import numpy as np
import matplotlib.pyplot as plt

from sklearn import cluster, datasets
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler

from bokeh.plotting import Figure, gridplot
from bokeh.palettes import Spectral6
from bokeh.models.widgets import VBox, HBox, Select, TextInput, List, Component, Instance, Slider

from bokeh.models import ColumnDataSource
from bokeh.io import curdoc

import inspect


class VisualDataClustering:
    """Methods:
     - __init__: takes Data to plot, returns VBox object
     - get_plots: returns a bokeh object that can be displayed
     - 
     - 
    """

    k_means_slider = Instance(Slider)
    DBSCAN_slider = Instance(Slider)
    birch_slider = Instance(Slider)
    colors = [[]]*4
    source = [[]]*4

    def __init__(self, X, y): 
        """Initializes the data viewer

        args:
         - X: array of x,y data to plot
         - y: coloring data, length equal to number of (x,y) data pairs
        """

        # normalize dataset for easier parameter selection
        self.X = StandardScaler().fit_transform(X)

        # estimate bandwidth for mean shift
        self.bandwidth = cluster.estimate_bandwidth(X, quantile=0.3)

        for i in range(4):
            self.colors[i] = [Spectral6[f] for f in y]
            self.source[i] = ColumnDataSource(data=dict(x=self.X[:, 0], y=self.X[:, 1], colors=self.colors[i]))

        # SET UP PLOT
        algorithm = 'Select Algorithm'
        tools = 'pan,wheel_zoom,box_select,reset'

        widgets_to_check_for_updates = []

        #KMeans
        plot1 = Figure(plot_width=400, plot_height=400, title="KMeans", 
                      title_text_font_size='10pt', tools=tools)
        plot1.circle('x', 'y', fill_color='colors', line_color=None, source=self.source[0])
        self.k_means_slider = Slider(start=1, end=20, value=2, step=1)
        widgets_to_check_for_updates.append(self.k_means_slider)

        plot2 = Figure(plot_width=400, plot_height=400, title="DBSCAN",
                      title_text_font_size='10pt', tools=tools)
        plot2.circle('x', 'y', fill_color='colors', line_color=None, source=self.source[1])
        self.DBSCAN_slider = Slider(start=0.01, end=1.0, step=0.01, value=0.2)
        widgets_to_check_for_updates.append(self.DBSCAN_slider)

        plot3 = Figure(plot_width=400, plot_height=400, title="Birch",
                      title_text_font_size='10pt', tools=tools)
        plot3.circle('x', 'y', fill_color='colors', line_color=None, source=self.source[2])
        self.birch_slider = Slider(start=1, end=20, value=2, step=1)
        widgets_to_check_for_updates.append(self.birch_slider)

        plot4 = Figure(plot_width=400, plot_height=400, title="Mean Shift",
                      title_text_font_size='10pt', tools=tools)
        plot4.circle('x', 'y', fill_color='colors', line_color=None, source=self.source[3])

        for widget in widgets_to_check_for_updates:
            widget.on_change('value', self.update_data)

        # SET UP LAYOUT
        self.plots = HBox(children=[VBox(children=[plot1, self.k_means_slider, plot3, self.birch_slider]), 
            VBox(children=[plot2, self.DBSCAN_slider, plot4])])
        # add to document

        self.update_data('value', 0, 0)
        
    def get_plots(self):
        return self.plots


    # SET UP CALLBACKS
    def update_data(self, attrname, old, new):

        #store the models here
        models = [cluster.MiniBatchKMeans(n_clusters = self.k_means_slider.value),
        cluster.DBSCAN(eps=self.DBSCAN_slider.value),
        cluster.Birch(n_clusters=self.birch_slider.value),
        cluster.MeanShift(bandwidth=self.bandwidth, bin_seeding=True)]
        #AgglomerativeClustering

        assert len(models)==4

        for model in models:
            model.fit(self.X)

        for i in range(4):
            if hasattr(model, 'labels_'):
                    y_pred = models[i].labels_.astype(np.int)
            else:
                    y_pred = models[i].predict(self.X)

            self.colors[i] = [Spectral6[f%6] for f in y_pred]

            self.source[i].data['colors'] = self.colors[i]


# SET UP DATA
np.random.seed(0)
n_samples = 1500

# Noisy circles dataset
X, y = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05)
#X = data point, y = group of each data point


curdoc().add_root(VBox(children=[VisualDataClustering(X, y).get_plots()]))