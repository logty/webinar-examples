import numpy as np
import matplotlib.pyplot as plt

from sklearn import cluster, datasets
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler

from bokeh.plotting import Figure
from bokeh.palettes import Spectral6
from bokeh.models.widgets import VBox, HBox, Select, TextInput, List, Component, Instance

from bokeh.models import ColumnDataSource
from bokeh.io import curdoc

import inspect

# SET UP DATA
np.random.seed(0)
n_samples = 1500

# Noisy circles dataset
X, y = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05)
# normalize dataset for easier parameter selection
X = StandardScaler().fit_transform(X)
# estimate bandwidth for mean shift
bandwidth = cluster.estimate_bandwidth(X, quantile=0.3)
# connectivity matrix for structured Ward
connectivity = kneighbors_graph(X, n_neighbors=10, include_self=False)
# make connectivity symmetric
connectivity = 0.5 * (connectivity + connectivity.T)
colors = [Spectral6[0] for i in y]
source = ColumnDataSource(data=dict(x=X[:, 0], y=X[:, 1], colors=colors))

# SET UP PLOT
algorithm = 'Select Algorithm'
tools = 'pan,wheel_zoom,box_select,reset'
plot = Figure(plot_width=400, plot_height=400, title=None,
              title_text_font_size='10pt', tools=tools)
plot.circle('x', 'y', fill_color='colors', line_color=None, source=source)

# SET UP WIDGET
'''clustering_algorithms= ['Select Algorithm',
    'MiniBatchKMeans', 'AffinityPropagation', 'MeanShift',
    'SpectralClustering', 'Ward', 'AgglomerativeClustering',
    'DBSCAN', 'Birch']'''
clustering_algorithms = []
clustering_name_to_obj_dict = {}
for name, obj in inspect.getmembers(cluster):
    if inspect.isclass(obj):
        clustering_algorithms += [name]
        clustering_name_to_obj_dict[name] = obj


dropdown = Select(value='Select Algorithm', options=clustering_algorithms)

clustering_options = Instance(VBox)
temp_chillun = []
options = List(Instance(Component))

# SET UP CALLBACKS
def update_data(attrname, old, new):

    # Get the drop down values
    algorithm = dropdown.value
    global X

    # Generate the new colors:
    if algorithm=='MiniBatchKMeans':
        model = cluster.MiniBatchKMeans(n_clusters=2)
    elif algorithm=='AffinityPropagation':
        model = cluster.AffinityPropagation(damping=.9, preference=-200)
    elif algorithm=='MeanShift':
        model = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
    elif algorithm=='SpectralClustering':
        model = cluster.SpectralClustering(n_clusters=2,
                                          eigen_solver='arpack',
                                          affinity="nearest_neighbors")
    elif algorithm=='Ward':
        model = cluster.AgglomerativeClustering(n_clusters=2, linkage='ward',
                                           connectivity=connectivity)
    elif algorithm=='AgglomerativeClustering':
        model = cluster.AgglomerativeClustering(
            linkage="average", affinity="cityblock", n_clusters=2,
            connectivity=connectivity)
    elif algorithm=='Birch':
        model = cluster.Birch(n_clusters=2)
    elif algorithm=='DBSCAN':
        model = cluster.DBSCAN(eps=.2)
    else:
        print('No Algorithm selected')

    
    argspec = inspect.getargspec(clustering_name_to_obj_dict[algorithm].__init__)
    for i in range(1,len(argspec.args)):
        temp_chillun.append(TextInput(title=argspec.args[i], value=str(argspec.defaults[i-1])))

    clustering_options.children = temp_chillun

    print "/////////////////////////////////////////////////////"
    print argspec
    print argspec.args
    print temp_chillun
    print "/////////////////////////////////////////////////////"

    #clustering_options = HBox(children=[options])

    model.fit(X)

    if hasattr(model, 'labels_'):
            y_pred = model.labels_.astype(np.int)
    else:
            y_pred = model.predict(X)

    colors = [Spectral6[i] for i in y_pred]

    source.data['colors'] = colors
    plot.title = algorithm

dropdown.on_change('value', update_data)

# SET UP LAYOUT
clustering_options = VBox(*temp_chillun)
inputs = HBox(children=[dropdown])
plots = HBox(children=[plot])
# add to document
curdoc().add_root(VBox(children=[inputs, plots, clustering_options]))
update_data()