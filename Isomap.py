import numpy as np
import numpy.linalg as npl
import pandas as pd
import sys
import seaborn as sns
import matplotlib.pyplot as plt

hole = False
if hole:
	d = pd.read_csv("hw5_swiss_roll/swiss_roll_hole.txt", names=['x', 'y', 'z'], sep='  |   ', engine='python')
	cmap = "Greens"
	color = 'g'
	title = 'swiss_role_hole'
else:
	d = pd.read_csv("hw5_swiss_roll/swiss_roll.txt", names=['x', 'y', 'z'], sep='  |   ', engine='python')
	cmap = "Reds"
	color = 'r'
	title = 'swiss_role'

data = np.column_stack((d.x, d.y, d.z))
dim = 2
nbrs = 5


''' takes in 
data: an array of points
dim : the dimension to project to
nbrs: number of neighbors you want in your graph
I have used the swiss role data here, but it works for any data (I hope)
just this function should be sufficient for 2iii
'''

def isomap(data, dim, nbrs):
	def mds(D, dimensions = dim):
	    E = (-0.5 * D**2)

	    # get means of rows and columns 
	    # need to center the matrix E before applying SVD to get the eigvals and eigvecs
	    rowmean = np.mat(np.mean(E,1))
	    columnmean = np.mat(np.mean(E,0))

	    F = np.array(E - np.transpose(rowmean) - columnmean + np.mean(E))

	    [U, S, V] = npl.svd(F)

	    Y = U * np.sqrt(S)

	    return Y[:,0:dimensions]


	def create_G(X, nbrs):
		from sklearn.neighbors import NearestNeighbors, kneighbors_graph

		neighbors = NearestNeighbors(nbrs)
		neighbors.fit(data)

		graph = kneighbors_graph(neighbors, nbrs)

		from sklearn.utils.graph import graph_shortest_path
		G = graph_shortest_path(graph, method='D', directed=False)

		return G

	G = create_G(data, nbrs)
	return mds(G)

projection = isomap(data, dim, nbrs)