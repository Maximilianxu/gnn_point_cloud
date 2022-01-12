import torch
from util import *
import torch.nn as nn
import torch_scatter
import dgl
import gnn_ext

def multi_linear_network_fn(Ks, is_logits=False):
  linears = []
  for i in range(1, len(Ks) - 1):
    linears.extend([nn.Linear(Ks[i - 1], Ks[i]), nn.ReLU()])
  
  linears.extend([nn.Linear(Ks[-2], Ks[-1])] + ([] if is_logits else [nn.ReLU()]))
  return nn.Sequential(*linears)

# set_features = torch.zeros(3, 8192, device=torch.device("cuda:0"))

def max_aggregation_fn(features, set_indices, keypoints_len):
  """
  features: [N, D]
  set_indices: [N], the group index of each feature belong to, used to scatter
  keypoints_len: K, there are K distinct group indices in set_indices
  """
  # global set_features
  # LOGD("feat size:", features.size(), "set ind size:", set_indices.size())
  set_indices = set_indices.unsqueeze(0).expand(features.shape[1], -1) # [N] -> [D, N]
  # set_features: [D, K], used to scatter N points to K targets
  set_features = torch.zeros(features.shape[-1], keypoints_len, device=torch.device("cuda:0"))
  # LOGD("expect set feat size:", [features.shape[-1]] + [keypoints_len])
  # the torch scatter 's input tensors should have size: N1 * N2, where N2 is the scatter apply dimension
  # for features N * D, we need to scatter D among N samples, N is the scatter dimension, thus, permute is needed
  # LOGD(features.size(), set_features.size(), set_indices.size())
  final_features, _ = torch_scatter.scatter_max(features.permute(1, 0), set_indices, out=set_features)
  final_features = final_features[:, :keypoints_len]
  #LOGD(final_features.size())
  final_features = final_features.permute(1, 0).contiguous() # permute the result
  return final_features

class PointSetPooling(nn.Module):
  def __init__(self, point_MLP_depth_list=[4, 32, 64, 128, 300], output_MLP_depth_list=[300, 300, 300], **kwargs) -> None:
    super().__init__()
    self.point_linears = multi_linear_network_fn(point_MLP_depth_list)
    self.out_linears = multi_linear_network_fn(output_MLP_depth_list)
  
  def forward(self, features, coordinates, keypoints, set_indices):
    """
      keypoints: a [K, 1] tensor. Indices of K keypoints.
      set_indices: a [2, S] tensor. S pairs of (point_index, set_index).
      i.e. (i, j) indicates point[i] belongs to the point set created by
      grouping around keypoint[j].
    """
    # ps means point set
    ps_features = features[set_indices[0, :]]
    ps_coordinates = coordinates[set_indices[0, :]]# gather source node coords
    ps_keypoints = keypoints[set_indices[1, :]] # gather keypoints indices
    ps_keypoint_coordinates = coordinates[ps_keypoints[:, 0]]

    ps_coordinates = ps_coordinates - ps_keypoint_coordinates
    ps_features = torch.cat([ps_features, ps_coordinates], dim=1)

    out_ps_features = self.point_linears(ps_features)
    # set_indices[:, 1]: [S, 1], out_ps_features: [S, D], len(key_points) = K, S features map to K keypoints
    agg_features = max_aggregation_fn(out_ps_features, set_indices[1, :], len(keypoints))
    out_agg_features = self.out_linears(agg_features)
    return out_agg_features

# we use pyg to implement graphnetAutoCenter
from torch_geometric.nn import MessagePassing
class GraphNetAutoCenter(MessagePassing):
  def __init__(self, auto_offset=True, auto_offset_MLP_depth_list=[300, 64, 3], edge_MLP_depth_list=[303, 300, 300], 
    update_MLP_depth_list=[300, 300, 300], **kwargs):
    super().__init__()
    self.auto_offset = auto_offset
    self.auto_offset_fn = multi_linear_network_fn(auto_offset_MLP_depth_list, is_logits=True)
    self.edge_feature_fn = multi_linear_network_fn(edge_MLP_depth_list)
    self.update_fn = multi_linear_network_fn(update_MLP_depth_list, is_logits=True)
  
  def forward(self, *args):
    if backend_framework == "torch":
      return self.torch_forward(*args)
    elif backend_framework == "pyg":
      return self.pyg_forward(*args)
    elif backend_framework == "dgl":
      assert backend_framework == "dgl"
      return self.dgl_forward(*args)
    elif backend_framework == "ext":
      return self.ext_forward(*args)
  
  def dgl_forward(self, graph):
    def update_udf(nodes):
      return {"x": nodes.data["x"] + self.update_fn(max_aggregation_fn(edge_feats, graph.edges()[1], graph.key.size(0)))}

    with graph.local_scope():
      src_coords = graph.ndata["p"][graph.edges()[0]]
      if self.auto_offset:
        offset = self.auto_offset_fn(graph.ndata["x"])
        ao_coords = graph.ndata["p"] + offset
      diff_coords = src_coords - ao_coords[graph.edges()[1]]
      edge_feats = torch.cat([graph.ndata["x"][graph.edges()[0]], diff_coords], dim=1)
      edge_feats = self.edge_feature_fn(edge_feats)
      graph.apply_nodes(update_udf)
      return graph.ndata["x"]
  
  def ext_forward(self, x, edge_index, keys, coords, row_pointers, column_index, degrees, 
    part_ptr, part_to_nodes, part_size=32, dim_worker=30, warp_per_block=32):

    # the two parts can run in parallel, if we use two GPUs, or two streams
    # LOGD(x.size(), coords.size())
    src_feats = self.edge_feature_fn(torch.cat([x, coords], dim=1))
    feats, = gnn_ext.sage_forward(src_feats, row_pointers, column_index, degrees, part_ptr, part_to_nodes, 
      part_size, dim_worker, warp_per_block, "max")
    
    source_coordinates = coords[edge_index[0, :], :]
    if self.auto_offset:
      offset = self.auto_offset_fn(x)
      ao_coordinates = coords + offset
    diff_coordinates = source_coordinates - ao_coordinates[edge_index[1, :], :]
    # aggr_coords = max_aggregation_fn(diff_coordinates, edge_index[1, :], keys.size(0))
    # cat_feats = torch.cat([feats, aggr_coords], dim=1)
    
    upd_feats = self.update_fn(feats)
    upd_feats += x
    return upd_feats

  def torch_forward(self, x, coordinates, keypoints, edge_index):
    # use torch native API, not pyg
    source_coordinates = coordinates[edge_index[0, :], :]
    if self.auto_offset:
      offset = self.auto_offset_fn(x)
      ao_coordinates = coordinates + offset
    
    diff_coordinates = source_coordinates - ao_coordinates[edge_index[1, :], :]
    edge_features = torch.cat([x[edge_index[0, :]], diff_coordinates], dim=1) # [E, D+3]
    # LOGD("calc edge feat")
    edge_features = self.edge_feature_fn(edge_features)
    # LOGD("mess ret")
    edge_features = max_aggregation_fn(edge_features, edge_index[1, :], keypoints.size(0))
    upd_features = self.update_fn(edge_features) 
    upd_features += x # residual
    return upd_features
  
  def pyg_forward(self, x, coordinates, keypoints, edge_index):
    """
    x: [N, D], node features
    edge_index: [2, E], edge sources/dests
    coordinates: [N, 3], 3 is 3-D dimension for point cloud
    keypoints: there are (num of keypoints) distinct target nodes in edge_index
    """
    source_coordinates = coordinates[edge_index[0, :], :]
    if self.auto_offset:
      offset = self.auto_offset_fn(x)
      ao_coordinates = coordinates + offset
    
    diff_coordinates = source_coordinates - ao_coordinates[edge_index[1, :], :]
    return self.propagate(edge_index, x=x, diff_coordinates=diff_coordinates, num_keypoints=keypoints.size(0))

  def message(self, x_j, diff_coordinates):
    """
      x_j: [E, D], source node features
      s/t_coordinates: [E, 3], 3 is 3-D dimension for point cloud
      keypoints: [K, 1], indices of keypoints
    """
    edge_features = torch.cat([x_j, diff_coordinates], dim=1) # [E, D+3]
    edge_features = self.edge_feature_fn(edge_features)
    return edge_features

  def aggregate(self, edge_features, edge_index, num_keypoints):
    # assert torch.unique(edge_index[1, :]).size(0) == num_keypoints
    out_edge_feat = max_aggregation_fn(edge_features, edge_index[1, :], num_keypoints)
    return out_edge_feat

  def update(self, aggr_features, x): # fancy param inspector
    upd_features = self.update_fn(aggr_features) # pyg yes!
    upd_features += x # residual
    return upd_features

class ClassAwarePredictor(nn.Module):
  def __init__(self, cls_MLP_depth_lst, box_encoding_method, box_MLP_depth_lst, **kwargs) -> None:
    super().__init__()
    self.cls_fn = multi_linear_network_fn(cls_MLP_depth_lst, is_logits=True)
    self.loc_fns = nn.ModuleList()
    self.num_classes = cls_MLP_depth_lst[-1]
    self.box_encoding_len = get_box_encoding_len(box_encoding_method)
    for _ in range(self.num_classes):
      self.loc_fns.append(multi_linear_network_fn(box_MLP_depth_lst + [self.box_encoding_len], is_logits=True))

  def forward(self, x):
    logits = self.cls_fn(x)
    box_encodings_lst = []
    for loc_fn in self.loc_fns:
      box_encodings = loc_fn(x)
      box_encodings_lst.append(box_encodings)
    box_encodings = torch.cat(box_encodings_lst, dim=1)
    return logits, box_encodings

def get_model_fn(model_type):
  layer_type_dict = {
      "scatter_max_point_set_pooling": PointSetPooling,
      "scatter_max_graph_auto_center_net": GraphNetAutoCenter,
      "classaware_predictor": ClassAwarePredictor
    }
  return layer_type_dict[model_type]

class MultiLayerFastLocalGraph(nn.Module):
  def __init__(self, model_config) -> None:
    super().__init__()
    # self.box_encoding_len = get_box_encoding_len(model_config["box_encoding_method"])
    self.layers = nn.ModuleList()
    layer_configs = model_config["layer_configs"]
    for lc in layer_configs[:-1]:
      layer = get_model_fn(lc["type"])(**lc["kwargs"])
      self.layers.append(layer)
    self.layer_configs = layer_configs
    last_pred_cfg = layer_configs[-1]
    self.predictor = get_model_fn(last_pred_cfg["type"])(**last_pred_cfg["kwargs"])
  
  def __get_layer_features__(self, feat, lens, graph_level, dim=0):
    start = lens[graph_level - 1] if graph_level > 0 else 0
    slices = [slice(None)] * len(feat.size())
    slices[dim] = slice(start, start + lens[graph_level])
    layer_feat = feat[slices]
    assert layer_feat.size(dim) == lens[graph_level]
    return layer_feat
  
  def forward(self, *args):
    if backend_framework == "dgl":
      return self.dgl_forward(*args)
    elif backend_framework == "pyg" or backend_framework == "torch":
      return self.pyg_forward(*args)
    elif backend_framework == "ext":
      return self.ext_forward(*args)
  
  def ext_forward(self, graphs):
    for i in range(len(self.layers)):
      graph_level = self.layer_configs[i]["graph_level"]
      layer = self.layers[i]
      sg = graphs[graph_level]
      for j in range(len(sg)):
        sg[j] = sg[j].cuda()
      # this edge_index shape is 2xE
      if isinstance(layer, GraphNetAutoCenter):
        feat = layer(*sg)
      else:
        feat, edges, keys, coords = sg[:4]
        feat = layer(feat, coords, keys, edges)
      if i < len(self.layers) - 1:
        ngl = self.layer_configs[i + 1]["graph_level"]
        graphs[ngl][0] = feat
    
    return self.predictor(feat)

  def dgl_forward(self, graphs, cuda=True):
    dev = torch.device("cuda:0") if cuda and torch.cuda.is_available() else torch.device("cpu")
    for i in range(len(self.layers)):
      graph_level = self.layer_configs[i]["graph_level"]
      layer = self.layers[i]
      sg = graphs[graph_level]
      sg = sg.to(dev)
      features = sg.ndata["x"] # if i == 0, then this is the init features; otherwise, it has been set at line 216
      if isinstance(layer, GraphNetAutoCenter):
        features = layer(sg) 
      else:
        features = layer(features, sg.ndata["p"], sg.key, torch.stack(sg.edges()))
      if i < len(self.layers) - 1:
        ngl = self.layer_configs[i + 1]["graph_level"]
        graphs[ngl] = graphs[ngl].to(dev)
        graphs[ngl].ndata["x"] = features
      
    logits, box_encodings = self.predictor(features)
    return logits, box_encodings

  def pyg_forward(self, batch, cuda=True):
    features, coords, keypoints, edges, e_lens, c_lens, k_lens = batch.x, batch.pos, \
      batch.keypoints, batch.edge_index, batch.e_lens, batch.c_lens, batch.k_lens
    dev = torch.device("cuda:0") if cuda and torch.cuda.is_available() else torch.device("cpu")
    features = features.to(dev)
    for i in range(len(self.layers)):

      graph_level = self.layer_configs[i]["graph_level"]
      layer_coords = self.__get_layer_features__(coords, c_lens, graph_level)
      layer_keypoints = self.__get_layer_features__(keypoints, k_lens, graph_level)
      layer_edges = self.__get_layer_features__(edges, e_lens, graph_level, dim=1)
      layer = self.layers[i]
      layer_coords = layer_coords.to(dev)
      layer_edges = layer_edges.to(dev)
      features = layer(features, layer_coords, layer_keypoints, layer_edges)
      del layer_coords, layer_edges
      torch.cuda.empty_cache()

    logits, box_encodings = self.predictor(features)
    return logits, box_encodings
