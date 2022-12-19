import gudhi
from gudhi.datasets.generators import points
from utils import calculate_distance_matrix
from scipy.spatial import distance

torus_dataset = points.torus(n_samples = 100, dim=3)
print(torus_dataset[:5])

X = calculate_distance_matrix(torus_dataset, distance.euclidean)
rips_complex = gudhi.RipsComplex(X)
simplex_tree = rips_complex.create_simplex_tree()


simplex_tree.compute_persistence()
print(simplex_tree.dimension())
print(simplex_tree.persistence_pairs())
for simplex, _ in simplex_tree.get_simplices():
    print(simplex)