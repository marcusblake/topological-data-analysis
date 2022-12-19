import numpy as np

"""
Calculates distance matrix from data points given a user specified metric

Input:
    data - Data points assumed to be from some metric space
    metric - Distance function to use to calculate the distance between points

Output:
    n x n matrix that contains pairwise distances between points
"""
def calculate_distance_matrix(data, metric):
    n = len(data)
    distances = np.zeros((n,n))

    for i in range(n):
        for j in range(i,n):
            distance = metric(data[i], data[j])
            distances[i][j], distances[j][i] = distance, distance
    
    return distances
