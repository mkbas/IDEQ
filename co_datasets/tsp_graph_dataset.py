"""TSP (Traveling Salesman Problem) Graph Dataset"""

import numpy as np
import torch

from sklearn.neighbors import KDTree
from torch_geometric.data import Data as GraphData


class TSPGraphDataset(torch.utils.data.Dataset):
  def __init__(self, data_file, sparse_factor=-1):
    self.data_file = data_file
    self.sparse_factor = sparse_factor
    self.file_lines = open(data_file).read().splitlines()
    print(f'Loaded "{data_file}" with {len(self.file_lines)} lines')

  def __len__(self):
    return len(self.file_lines)

  def get_example(self, idx):
    # Select sample
    line = self.file_lines[idx]
    # Clear leading/trailing characters
    line = line.strip()

    # Extract points
    points = line.split(' output ')[0]
    points = points.split(' ')
    points = np.array([[float(points[i]), float(points[i + 1])] for i in range(0, len(points), 2)])
    # Extract tour
    tour = line.split(' output ')[1]
    tour = tour.split(' ')
    tour = np.array([int(t) for t in tour])
    tour -= 1

    return points, tour

  def __getitem__(self, idx):
    #idx=0

    points, tour = self.get_example(idx)
    if self.sparse_factor <= 0:
      # Return a densely connected graph
      adj_matrix = np.zeros((points.shape[0], points.shape[0]))
      #np.save('GT',adj_matrix)
      for i in range(tour.shape[0] - 1):
        adj_matrix[tour[i], tour[i + 1]] = 1
      # return points, adj_matrix, tour
      return (
          torch.LongTensor(np.array([idx], dtype=np.int64)),
          torch.from_numpy(points).float(),
          torch.from_numpy(adj_matrix).float(),
          torch.from_numpy(tour).long(),
      )
    else:
      # Return a sparse graph where each node is connected to its k nearest neighbors
      # k = self.sparse_factor
      sparse_factor = self.sparse_factor
      kdt = KDTree(points, leaf_size=30, metric='euclidean')
      dis_knn, idx_knn = kdt.query(points, k=sparse_factor, return_distance=True)

      edge_index_0 = torch.arange(points.shape[0]).reshape((-1, 1)).repeat(1, sparse_factor).reshape(-1)
      edge_index_1 = torch.from_numpy(idx_knn.reshape(-1))

      edge_index = torch.stack([edge_index_0, edge_index_1], dim=0)

      tour_edges = np.zeros(points.shape[0], dtype=np.int64)
      tour_edges[tour[:-1]] = tour[1:]
      tour_edges = torch.from_numpy(tour_edges)
      tour_edges = tour_edges.reshape((-1, 1)).repeat(1, sparse_factor).reshape(-1)
      tour_edges = torch.eq(edge_index_1, tour_edges).reshape(-1, 1)
      graph_data = GraphData(x=torch.from_numpy(points).float(),
                             edge_index=edge_index,
                             edge_attr=tour_edges)

      point_indicator = np.array([points.shape[0]], dtype=np.int64)
      edge_indicator = np.array([edge_index.shape[1]], dtype=np.int64)
      #np.save('GTsparse',edge_index)
      return (
          torch.LongTensor(np.array([idx], dtype=np.int64)), # [N, 1]
          graph_data,
          torch.from_numpy(point_indicator).long(), # [B, N, 2]
          torch.from_numpy(edge_indicator).long(), # [B, N, N]
          torch.from_numpy(tour).long(), # [B, N+1]
      )

class TSPlibGraphDataset(torch.utils.data.Dataset):
  def __init__(self, data_file, sparse_factor=-1):
    self.data_file = data_file
    self.sparse_factor = sparse_factor
    self.file_lines = open(data_file).read().splitlines()
    print(f'Loaded "{data_file}" with {len(self.file_lines)} lines')

  def __len__(self):
    return len(self.file_lines)

  def get_example(self, idx):
    # Select sample
    line = self.file_lines[idx]
    # Clear leading/trailing characters
    line = line.strip()

    # Extract points
    points = line.split(' output ')[0]
    points = points.split(' ')
    points = np.array([[float(points[i]), float(points[i + 1])] for i in range(0, len(points), 2)])
    # Extract tour
    tour = line.split(' output ')[1]
    tour = tour.split(' ')
    tour = np.array([int(t) for t in tour])
    tour -= 1

    return points, tour

  def __getitem__(self, idx):
    #idx=0

    points, tour = self.get_example(idx)
    # Return a sparse graph where each node is connected to its k nearest neighbors
    # k = self.sparse_factor
    sparse_factor = self.sparse_factor
    kdt = KDTree(points, leaf_size=30, metric='euclidean')
    dis_knn, idx_knn = kdt.query(points, k=sparse_factor, return_distance=True)

    edge_index_0 = torch.arange(points.shape[0]).reshape((-1, 1)).repeat(1, sparse_factor).reshape(-1)
    edge_index_1 = torch.from_numpy(idx_knn.reshape(-1))

    edge_index = torch.stack([edge_index_0, edge_index_1], dim=0)

    tour_edges = np.zeros(points.shape[0], dtype=np.int64)
    tour_edges[tour[:-1]] = tour[1:]
    tour_edges = torch.from_numpy(tour_edges)
    tour_edges = tour_edges.reshape((-1, 1)).repeat(1, sparse_factor).reshape(-1)
    tour_edges = torch.eq(edge_index_1, tour_edges).reshape(-1, 1)
    graph_data = GraphData(x=torch.from_numpy(points).float(),
                           edge_index=edge_index,
                           edge_attr=tour_edges)

    point_indicator = np.array([points.shape[0]], dtype=np.int64)
    edge_indicator = np.array([edge_index.shape[1]], dtype=np.int64)
    #np.save('GTsparse',edge_index)
    return (
        torch.LongTensor(np.array([idx], dtype=np.int64)), # [N, 1]
        graph_data,
        torch.from_numpy(point_indicator).long(), # [B, N, 2]
        torch.from_numpy(edge_indicator).long(), # [B, N, N]
        torch.from_numpy(tour).long(), # [B, N+1]
    )

class TSPGraphDatasetinv2opt(torch.utils.data.Dataset):
  def __init__(self, data_file, sparse_factor=-1, n2opt=2):
    self.data_file = data_file
    self.sparse_factor = sparse_factor
    self.file_lines = open(data_file).read().splitlines()
    self.n2opt=n2opt
    print(f'Loaded "{data_file}" with {len(self.file_lines)} linesusing 0-{n2opt} backward two opt moves')

  def __len__(self):
    return len(self.file_lines)


  @classmethod
  def revtwoopt(cls,tour,n_nodes):
      valid=False
      while valid==False:
          i=np.random.randint(low=1,high=n_nodes-2)
          j=np.random.randint(low=i+1, high=n_nodes-1)
          new_tour=tour[:i]+tour[j:-len(tour) + i - 1:-1]+tour[j+1:]
          valid=True ########### !!!!!!!!!!!
      return new_tour

  def get_example(self, idx):
    # Select sample
    line = self.file_lines[idx]
    # Clear leading/trailing characters
    line = line.strip()

    # Extract points
    points = line.split(' output ')[0]
    points = points.split(' ')
    points = np.array([[float(points[i]), float(points[i + 1])] for i in range(0, len(points), 2)])
    # Extract tour
    tour = line.split(' output ')[1]
    tour = tour.split(' ')
    tour = [int(t)-1 for t in tour]
    #tour -= 1

    return points, tour

  def __getitem__(self, idx):
    points, tour = self.get_example(idx)
    n=len(tour)-1
    n2opt=np.random.randint(self.n2opt+1)
    for _ in range(n2opt):
        tour=self.revtwoopt(tour, n)
    tour=np.array(tour)
    
    if self.sparse_factor <= 0:
      # Return a densely connected graph
      adj_matrix = np.zeros((points.shape[0], points.shape[0]))
      for i in range(tour.shape[0] - 1):
        adj_matrix[tour[i], tour[i + 1]] = 1
      # return points, adj_matrix, tour
      return (
          torch.LongTensor(np.array([idx], dtype=np.int64)),
          torch.from_numpy(points).float(),
          torch.from_numpy(adj_matrix).float(),
          torch.from_numpy(tour).long(),
      )
    else:
      # Return a sparse graph where each node is connected to its k nearest neighbors
      # k = self.sparse_factor
      sparse_factor = self.sparse_factor
      kdt = KDTree(points, leaf_size=30, metric='euclidean')
      dis_knn, idx_knn = kdt.query(points, k=sparse_factor, return_distance=True)

      edge_index_0 = torch.arange(points.shape[0]).reshape((-1, 1)).repeat(1, sparse_factor).reshape(-1)
      edge_index_1 = torch.from_numpy(idx_knn.reshape(-1))

      edge_index = torch.stack([edge_index_0, edge_index_1], dim=0)

      tour_edges = np.zeros(points.shape[0], dtype=np.int64)
      tour_edges[tour[:-1]] = tour[1:]
      tour_edges = torch.from_numpy(tour_edges)
      tour_edges = tour_edges.reshape((-1, 1)).repeat(1, sparse_factor).reshape(-1)
      tour_edges = torch.eq(edge_index_1, tour_edges).reshape(-1, 1)
      graph_data = GraphData(x=torch.from_numpy(points).float(),
                             edge_index=edge_index,
                             edge_attr=tour_edges)

      point_indicator = np.array([points.shape[0]], dtype=np.int64)
      edge_indicator = np.array([edge_index.shape[1]], dtype=np.int64)
      return (
          torch.LongTensor(np.array([idx], dtype=np.int64)),
          graph_data,
          torch.from_numpy(point_indicator).long(),
          torch.from_numpy(edge_indicator).long(),
          torch.from_numpy(tour).long(),
      )