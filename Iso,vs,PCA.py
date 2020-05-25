import numpy as np
import numpy.linalg as npl
import pandas as pd
import sys
import seaborn as sns
import matplotlib.pyplot as plt

hole = True
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


from sklearn.decomposition import PCA


pca = False
if pca:
	embedding = PCA(n_components=dim)
	projection = embedding.fit_transform(data)
else:
	projection = isomap(data, dim, nbrs)

from mpl_toolkits import mplot3d

def graph_projection(projection, color, title, save):
	x = [row[0] for row in projection]
	y = [row[1] for row in projection]
	zeros = np.zeros(len(x))

	fig = plt.figure()
	ax = plt.axes(projection='3d')

	ax.scatter3D(x, y, zeros, c=color)
	plt.title(title)
	plt.show()
	if save:
		fig.savefig('../ml/swiss_role_isomap_me.png')


graph_projection(projection, color, title, False)

def graph_3D(d, colormap, save=False):
	fig = plt.figure()
	ax = plt.axes(projection='3d')

	ax.scatter3D(d.x, d.y, d.z, c=d.z, cmap=colormap)
	plt.title('swiss_role')
	plt.show()
	if save:
		fig.savefig('../ml/swiss_role.png')

graph_3D(d, cmap)




