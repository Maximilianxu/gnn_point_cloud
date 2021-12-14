import torch
import gnn_ext
import numpy as np
import dgl
from scipy.sparse import *
# import gnn_ext_forward
# import torch

edge_index = [[0, 1, 2, 3], [1, 2, 1, 1]]

out_edge_index, mapping = gnn_ext.rabbit_reorder(edge_index)

print(out_edge_index, mapping)
t = torch.FloatTensor([1., 2.])
gnn_ext.test_torch(t)

def test_sage():
  src_edges = [0, 2]
  dst_edges = [1, 1]
  num_nodes = np.unique(src_edges + dst_edges).shape[0]
  print("num nodes:", num_nodes)
  hidden_dim = 3
  init_dim = 2
  g = dgl.graph((torch.IntTensor(src_edges), torch.IntTensor(dst_edges)))
  g.ndata["x"] = torch.FloatTensor([[i] * init_dim for i in range(num_nodes)])

  g = g.to(torch.device("cuda:0"))
  weight = torch.FloatTensor(np.ones((init_dim, hidden_dim))).cuda()

  print("init feat:", g.ndata["x"] @ weight)

  edge_vals = [1] * len(src_edges)
  coo = coo_matrix((edge_vals, (src_edges, dst_edges)), shape=(num_nodes, num_nodes))
   # CSC is what we need, CSR is not, since we take row index 
   # as node targets, column index as source, aggr from source to target
  csc = coo.tocsc()

  row_pointers = csc.indptr
  column_index = csc.indices
  print("row pointers:", row_pointers, column_index)

  # output degree
  degrees = (row_pointers[1:] - row_pointers[:-1]).tolist()
  # print(row_pointers, "\n", degrees)
  degrees = torch.sqrt(torch.FloatTensor(list(map(lambda x: 1 if x > 0 else 0, degrees))))

  # build parts, part_num, part_ptr, part_to_nodes
  part_size = 2
  # TODO: we need to reimpl this in C++
  # as we expect to aggr from src to target
  part_ptr, part_to_nodes = gnn_ext.build_part(part_size, torch.IntTensor(row_pointers))
  print(part_ptr, part_to_nodes)
  # expect this
  part_ptr = torch.IntTensor([0, 2])
  part_to_nodes = torch.IntTensor([1])
  print(part_ptr, part_to_nodes)
  # # exit()
  
  dim_worker = 32
  warp_per_block = 4
  
  out = gnn_ext.sage_forward(g.ndata["x"], weight, torch.IntTensor(row_pointers).cuda(), torch.IntTensor(column_index).cuda(), 
    degrees.cuda(), part_ptr.cuda(), part_to_nodes.cuda(), part_size, dim_worker, warp_per_block)
  print(out)

test_sage()

