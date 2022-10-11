# Preparing Data for Graph Construction
'''
As part of our larger data science goal for this workshop, we will be working with data reflecting the entire road network of Great Britain. We have as a starting point road data extracted into tabular csv format from official GML files. Ultimately, we would like to use cuGraph to perform GPU-accelerated graph analytics on this data, but in order to do so, we need to do some preprocessing to get it ready for graph creation.

In this notebook you will be learning additional cuDF data transformation techniques in a demonstration of prepping data for ingestion by cuGraph. Next, you will do a series of exercises to perform a similar transformation of the data for the creation of a graph with different edge weights.
'''

# Objectives
'''
By the time you complete this notebook you will be able to:
- Create a GPU-accelerated graph
- Perform GPU-accelerated dataframe merge operations with cuDF
'''

# Imports
'''
In addition to cudf, for this notebook we will also import cugraph, which we will use (after data preparation) to construct a GPU-accelerated graph. We also import networkx for a brief performance comparison later on.
'''

import cudf
import cugraph as cg

import networkx as nx

## Read Data
'''
In this notebook we will be working with two data sources that will help us create a graph of the UK's road networks.
'''
# UK Road Nodes
'''
The first data table describes the nodes in the road network: endpoints, junctions (including roundabouts), and points that break up a long stretch of curving road so that it can be mapped correctly (instead of as a straight line).

The coordinates for each point are in the OSGB36 format we explored earlier in section 1-05.
'''

road_nodes = cudf.read_csv('./data/road_nodes_1-06.csv')
road_nodes.head()
road_nodes.dtypes
road_nodes.shape
road_nodes['type'].unique()

# UK Road Edges
'''
The second data table describes road segments, including their start and end points, how long they are, and what kind of road they are.
'''

road_edges = cudf.read_csv('./data/road_edges_1-06.csv')
road_edges.head()
road_edges.dtypes
road_edges.shape
road_edges['type'].unique()
road_edges['form'].unique()

# Exercise: Make IDs Compatible
'''
Our csv files were derived from original GML files, and as you can see from the above, both road_edges['src_id'] and road_edges['dst_id'] contain a leading # character that road_nodes['node_id'] does not. To make the IDs compatible between the edges and nodes, use cuDF's string method .str.lstrip to replace the src_id and dst_id columns in road_edges with values stripped of the leading # characters.
'''
road_edges['src_id'] = road_edges['src_id'].str.lstrip('#')
road_edges['dst_id'] = road_edges['dst_id'].str.lstrip('#')
road_edges[['src_id', 'dst_id']].head()

# Data Summary
'''
Now that the data is cleaned we can see just how many roads and endpoints/junctions/curve points we will be working with, as well as its memory footprint in our GPU. The GPUs we are using can hold and analyze much larger graphs than this one!
'''

print(f'{road_edges.shape[0]} edges, {road_nodes.shape[0]} nodes')

# Building the Road Network Graph
'''
We don't have information on the direction of the roads (some of them are one-way), so we will assume all of them are two-way for simplicity. That makes the graph "undirected," so we will build a cuGraph Graph rather than a directed graph orDiGraph.

We initialize it with edge sources, destinations, and attributes, which for our data will be the length of the roads:
'''

G = cg.Graph()
%time G.from_cudf_edgelist(road_edges, source='src_id', destination='dst_id', edge_attr='length')

'''
Just as a point of comparison, we also construct the equivalent graph in NetworkX from the equivalent cleaned and prepped Pandas dataframe.
'''

road_edges_cpu = road_edges.to_pandas()
%time G_cpu = nx.convert_matrix.from_pandas_edgelist(road_edges_cpu, source='src_id', target='dst_id', edge_attr='length')

# Reindex road_nodes
'''
For efficient lookup later, we will reindex road_nodes to use the node_id as its index - remember, we will typically get results from the graph analytics in terms of node_ids, so this lets us easily pull other information about the nodes (like their locations). We then sort the dataframe on this new index:
'''
road_nodes = road_nodes.set_index('node_id', drop=True)
%time road_nodes = road_nodes.sort_index()
road_nodes.head()

# Analyzing the Graph
'''
Now that we have created the graph we can analyze the number of nodes and edges in it:
'''

G.number_of_nodes()
G.number_of_edges()

'''
Notice that the number of edges is slightly smaller than the number of edges in road_edges printed above--the original data came from map tiles, and roads that passed over the edge of a tile were listed in both tiles, so cuGraph deduplicated them. If we were creating a MultiGraph or MultiDiGraph--a graph that can have multiple edges in the same direction between nodes--then duplicates could be preserved.

We can also analyze the degrees of our graph nodes:
'''

deg_df = G.degree()

'''
In an undirected graph, every edge entering a node is simultaneously an edge leaving the node, so we expect the nodes to have a minimum degree of 2:
'''

deg_df['degree'].describe()[1:]

'''
You will spend more time using this GPU-accelerated graph later in the workshop.
'''

# Exercise: Construct a Graph of Roads with Time Weights
'''
For this series of exercises, you are going to construct and analyze a new graph of Great Britain's roads using the techniques just demonstrated, but this time, instead of using raw distance for the edges' weights, you will be using the time it will take to travel between the two nodes at a notional speed limit.

You will be beginning this exercise with the road_edges dataframe from earlier:
'''

road_edges.dtypes

# Road Type to Speed Conversion
'''
In order to calculate how long it should take to travel along a road, we need to know its speed limit. We will do this by utilizing road_edges['type'], along with rules for the speed limits for each type of road.

Here are the unique types of roads in our data:
'''

road_edges['type'].unique()

'''
And here is a table with assumptions about speed limits we can use for our conversion:
'''

# https://www.rac.co.uk/drive/advice/legal/speed-limits/
# Technically, speed limits depend on whether a road is in a built-up area and the form of carriageway,
# but we can use road type as a proxy for built-up areas.
# Values are in mph.
speed_limits = {'Motorway': 70,
               'A Road': 60,
               'B Road': 60,
               'Local Road': 30,
               'Local Access Road': 30,
               'Minor Road': 30,
               'Restricted Local Access Road': 30,
               'Secondary Access Road': 30}

# We begin by creating speed_gdf to store each road type with its speed limit:

speed_gdf = cudf.DataFrame()

speed_gdf['type'] = speed_limits.keys()
speed_gdf['limit_mph'] = [speed_limits[key] for key in speed_limits.keys()]
speed_gdf

# Next we add an additional column, limit_m/s, which for each road type will give us a measure of how fast one can travel on it in meters / second.
# We will have road distances in meters (m), so to get road distances in seconds (s), we need to multiply by meters/mile and divide by seconds/hour
# 1 mile ~ 1609.34 m
speed_gdf['limit_m/s'] = speed_gdf['limit_mph'] * 1609.34 / 3600
speed_gdf

# Step 1: Merge speed_gdf into road_edges
'''
cuDF provides merging functionality just like Pandas. Since we will be using values in road_edges to construct our graph, we need to merge speed_gdf into road_edges (similar to a database join). You can merge on the type column, which both of these dataframes share.
'''

%time road_edges = road_edges.merge(speed_gdf, on='type')

# Step 2: Add Length in Seconds Column
'''
You now need to calculate the number of seconds it will take to traverse a given road at the speed limit. This can be done by dividing a road's length in m by its speed limit in m/s. Perform this calculation on road_edges and store the results in a new column length_s.
'''
road_edges['length_s'] = road_edges['length'] / road_edges['limit_m/s']
road_edges['length_s'].head()

# Step 3: Construct the Graph
'''
Construct a cuGraph Graph called G_ex using the sources and destinations found in road_edges, along with length-in-seconds values for the edges' weights.
'''

G_ex = cg.Graph()
G_ex.from_cudf_edgelist(road_edges, source='src_id', destination='dst_id', edge_attr='length_s')