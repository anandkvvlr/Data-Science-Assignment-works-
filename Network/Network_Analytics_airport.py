### network analysis airport ###

import pandas as pd
import numpy as np
import networkx as nx 
import matplotlib.pyplot as plt

# Degree Centrality
G = pd.read_csv("G://network//Datasets_Network Analytics//connecting_routes.csv",sep=',')
E = pd.read_csv("G://network//Datasets_Network Analytics//flight_hault.csv",sep=',')
G = G.iloc[:, 0:8]

G.columns= ["flights", "ID", "main airport","main Airport ID","Destination","Destination ID","haults","machinary"]
G.head(10)
E.columns= ["Airport ID","Name","City","Country","IATA_FAA","ICAO","Latitude","Longitude","Altitude","Timezone","DST","Tz database timezone"]
E.head(10)

g = nx.Graph()

# only considering first 40 airport in list
g = nx.from_pandas_edgelist(G.iloc[0:40,:], source = 'main airport', target = 'Destination')

print(nx.info(g))

# Degree Centrality
b = nx.degree_centrality(g)  
print(b) 

pos = nx.spring_layout(g, k = 0.15)
nx.draw_networkx(g, pos, node_size = 25, node_color = 'blue')

# closeness centrality
closeness = nx.closeness_centrality(g)
print(closeness)

## Betweeness Centrality 
b = nx.betweenness_centrality(g) # Betweeness_Centrality
print(b)

## Eigen-Vector Centrality
evg = nx.eigenvector_centrality(g) # Eigen vector centrality
print(evg)


