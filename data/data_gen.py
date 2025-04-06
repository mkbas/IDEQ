import numpy as np
import os
from concorde.tsp import TSPSolver
import time 
import glob
import multiprocessing as mp


scale_f=1e6

def solve_tsp(nodes_coord):
    scale = scale_f
    solver = TSPSolver.from_data(nodes_coord[:, 0] * scale, nodes_coord[:, 1] * scale, norm="EUC_2D")
    solution = solver.solve(verbose=False)
    tour = solution.tour
    return tour


for num_samples, filename, num_nodes, seed  in zip([128000, 2048, 128000, 2048],
                                                   ["tsp500_train_concorde.txt","tsp500_test_concorde.txt","tsp1000_train_concorde.txt","tsp1000_test_concorde.txt"],
                                                   [500,500,1000,1000],
                                                   [1234,4321,1234,4321]):
    np.random.seed(seed)

    with open(filename, "w") as f:
      start_time = time.time()
      nodes_coord = np.random.random([num_samples,num_nodes, 2])
        
    
      with mp.Pool() as p:
        tours = p.map(solve_tsp, nodes_coord)
    
      for idx, tour in enumerate(tours):
        if (np.sort(tour) == np.arange(num_nodes)).all():
          f.write(" ".join(str(x) + str(" ") + str(y) for x, y in nodes_coord[idx]))
          f.write(str(" ") + str('output') + str(" "))
          f.write(str(" ").join(str(node_idx + 1) for node_idx in tour))
          f.write(str(" ") + str(tour[0] + 1) + str(" "))
          f.write("\n")

    end_time = time.time() - start_time

    print(f"Completed generation of {num_samples} samples of TSP-{num_nodes}.")
    print(f"Total time: {end_time / 60:.1f}m")
    print(f"Average time: {end_time / num_samples:.1f}s")
