import os
from pathlib import Path
from typing import Callable, Optional
import numpy as np
import torch
from torch_geometric.data import Data, Dataset
import cv2
import dgl
from dgl.data import DGLDataset
from util import *
import gnn_ext
from scipy.sparse import coo_matrix

class KittiHelper: # the meta class, used for both PyG dataset and DGL dataset
  def __init__(self, ds_config) -> None:
    super().__init__()
    root_dir = ds_config.get("root_dir").replace("~", str(Path.home()))
    is_training = ds_config.get("train", False)
    tmp_path = "train" if is_training else "testing"
    self._image_dir = os.path.join(root_dir, "image/{}/image_2".format(tmp_path))
    self._point_dir = os.path.join(root_dir, "velodyne/{}/velodyne".format(tmp_path))
    self._calib_dir = os.path.join(root_dir, "calib/{}/calib".format(tmp_path))
    self.is_training = is_training
    if is_training:
      self._label_dir = os.path.join(root_dir, "label/")
    else:
      self._label_dir = ""
    self._file_list = self._get_file_list(self._image_dir)
    self._num_classes = ds_config.get("num_classes", 8)
    self._difficulty = ds_config.get("difficulty", -100)
    self._num_samples = len(self._file_list)
    self._max_image_height = 376
    self._max_image_width = 1242
    self.mem_cache = MemCache(64)
    # other info about preprocessing
    self.__init_dataset__(ds_config)
  
  def __init_dataset__(self, config):
    bem = config["box_encoding_method"]
    self.box_encoding_len = get_box_encoding_len(bem)
    self.box_encoding_fn = get_box_encoding_fn(bem)
    self.box_decoding_fn = get_box_decoding_fn(bem)
    self.aug_fn = config.get("data_aug_configs", [])
    self.graph_gen_method = config["graph_gen_method"]
    self.config = config # for other detailed informations
  
  def _get_file_list(self, dir):
    file_list = [f.split(".")[0] for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
    file_list.sort()
    return file_list

  def _get_calib(self, idx):
    calib_file = os.path.join(self._calib_dir, self._file_list[idx]) + ".txt"
    with open(calib_file, "r") as f:
      calib = {}
      for line in f:
        fields = line.split(" ")
        matrix_name = fields[0].rstrip(":")
        matrix = np.array(fields[1:], dtype=np.float32)
        calib[matrix_name] = matrix
    calib['P2'] = calib['P2'].reshape(3, 4)
    calib['R0_rect'] = calib['R0_rect'].reshape(3,3)
    calib['Tr_velo_to_cam'] = calib['Tr_velo_to_cam'].reshape(3,4)
    R0_rect = np.eye(4)
    R0_rect[:3, :3] = calib['R0_rect']
    calib['velo_to_rect'] = np.vstack([calib['Tr_velo_to_cam'],[0,0,0,1]])
    calib['cam_to_image'] = np.hstack([calib['P2'][:, 0:3], [[0],[0],[0]]])
    calib['rect_to_cam'] = np.hstack([
        calib['R0_rect'],
        np.matmul(
            np.linalg.inv(calib['P2'][:, 0:3]), calib['P2'][:, [3]])])
    calib['rect_to_cam'] = np.vstack([calib['rect_to_cam'],
        [0,0,0,1]])
    calib['velo_to_cam'] = np.matmul(calib['rect_to_cam'],
        calib['velo_to_rect'])
    calib['cam_to_velo'] = np.linalg.inv(calib['velo_to_cam'])
    # senity check
    calib['velo_to_image'] = np.matmul(calib['cam_to_image'],
        calib['velo_to_cam'])
    assert np.isclose(calib['velo_to_image'],
        np.matmul(np.matmul(calib['P2'], R0_rect),
        calib['velo_to_rect'])).all()
    return calib

  def _get_velo_points(self, idx, xyz_range=None):
    point_file = os.path.join(self._point_dir, self._file_list[idx]) + ".bin"
    velo_data = np.fromfile(point_file, dtype=np.float32).reshape(-1, 4)
    velo_points = velo_data[:, :3]
    reflections = velo_data[:, [3]]
    if xyz_range is not None:
      x_range, y_range, z_range = xyz_range
      mask = (velo_points[:, 0] > x_range[0]) and (velo_points[:, 0] < x_range[1])
      mask = mask and (velo_points[:, 1] > y_range[0]) and (velo_points[:, 1] < y_range[1])
      mask = mask and (velo_points[:, 2] > z_range[0]) and (velo_points[:, 2] < z_range[1])
      return Points(xyz=velo_points[mask], attr=reflections[mask])
    return Points(xyz=velo_points, attr=reflections)

  def _velo_points_to_cam(self, velo_points, calib):
    cam_xyz = np.matmul(velo_points.xyz, np.transpose(calib["velo_to_cam"])[:3, :3].astype(np.float32))
    cam_xyz += np.transpose(calib["velo_to_cam"])[[3], :3].astype(np.float32)
    return Points(xyz=cam_xyz, attr=velo_points.attr)

  def _get_cam_points(self, idx, calib=None, xyz_range=None):
    # load velo points, convert to camera points
    velo_points = self._get_velo_points(idx, xyz_range)
    if calib is None:
      calib = self._get_calib(idx)
    cam_points = self._velo_points_to_cam(velo_points, calib)
    if self.config.get("downsample_by_voxel_size", None) is not None:
      cam_points = downsample_by_average_voxel(cam_points, self.downsample_voxel_size)
    return cam_points
  
  def _cam_points_to_image(self, points, calib):
    cam_points_xyz1 = np.hstack([points.xyz, np.ones([points.xyz.shape[0], 1])])
    img_points_xyz = np.matmul(cam_points_xyz1, np.transpose(calib["cam_to_image"]))
    img_points_xy1 = img_points_xyz / img_points_xyz[:, [2]]
    img_points = Points(img_points_xy1, points.attr)
    return img_points

  def _get_image(self, idx):
    image_file = os.path.join(self._image_dir, self._file_list[idx]) + ".png"
    return cv2.imread(image_file)

  def _rgb_to_cam_points(self, points, image, calib):
    img_points = self._cam_points_to_image(points, calib)
    rgb = image[np.int32(img_points.xyz[:, 1]), np.int32(img_points.xyz[:, 0]), ::-1].astype(np.float32) / 255.
    return Points(points.xyz, np.hstack([points.attr, rgb]))

  def _get_cam_points_in_image_with_rgb(self, idx, calib=None, xyz_range=None):
    if calib is None:
      calib = self._get_calib(idx)
    cam_points = self._get_cam_points(idx, calib=calib, xyz_range=xyz_range)
    front_cam_points_idx = cam_points.xyz[:, 2] > 0.1
    front_cam_points = Points(cam_points.xyz[front_cam_points_idx, :], \
      cam_points.attr[front_cam_points_idx, :])
    image = self._get_image(idx)
    height, width = image.shape[0], image.shape[1]
    img_points = self._cam_points_to_image(front_cam_points, calib)
    img_points_in_image_idx = np.logical_and.reduce(
      [img_points.xyz[:,0] > 0, img_points.xyz[:, 0] < width,
        img_points.xyz[:,1] > 0, img_points.xyz[:, 1] < height])
    cam_points_in_img = Points(xyz=front_cam_points.xyz[img_points_in_image_idx, :], \
      attr=front_cam_points.attr[img_points_in_image_idx, :])
    cam_points_in_img_with_rgb = self._rgb_to_cam_points(cam_points_in_img, image, calib)
    return cam_points_in_img_with_rgb
  
  def _get_label(self, idx, no_orientation=False):
    MIN_HEIGHT = [40, 25, 25]
    MAX_OCCLUSION = [0, 1, 2]
    MAX_TRUNCATION = [0.15, 0.3, 0.5]
    label_file = os.path.join(self._label_dir, self._file_list[idx]) + '.txt'
    label_list = []
    with open(label_file, 'r') as f:
      for line in f:
        label={}
        line = line.strip()
        if line == '':
          continue
        fields = line.split(' ')
        label['name'] = fields[0]
        # 0=visible 1=partly occluded, 2=fully occluded, 3=unknown
        label['truncation'] = float(fields[1])
        label['occlusion'] = int(fields[2])
        label['alpha'] =  float(fields[3])
        label['xmin'] =  float(fields[4])
        label['ymin'] =  float(fields[5])
        label['xmax'] =  float(fields[6])
        label['ymax'] =  float(fields[7])
        label['height'] =  float(fields[8])
        label['width'] =  float(fields[9])
        label['length'] =  float(fields[10])
        label['x3d'] =  float(fields[11])
        label['y3d'] =  float(fields[12])
        label['z3d'] =  float(fields[13])
        label['yaw'] =  float(fields[14])
        if len(fields) > 15:
          label['score'] =  float(fields[15])
        if self.difficulty > -1:
          if label['truncation'] > MAX_TRUNCATION[self.difficulty]:
            continue
          if label['occlusion'] > MAX_OCCLUSION[self.difficulty]:
            continue
          if (label['ymax'] - label['ymin']) < MIN_HEIGHT[self.difficulty]:
            continue
        label_list.append(label)
    return label_list
  
  def assign_classaware_label_to_points(self, labels, xyz, expend_factor):
    """Assign class label and bounding boxes to xyz points. """
    # print("Yes, I am Here !!!!!!!!!!!!!!!!!!!!!")
    assert self.num_classes == 8
    num_points = xyz.shape[0]
    assert num_points > 0, "No point No prediction"
    assert xyz.shape[1] == 3
    # define label map
    label_map = {'Background': 0, 'Car': 1, 'Pedestrian': 3, 'Cyclist': 5, 'DontCare': 7}
    # by default, all points are assigned with background label 0.
    cls_labels = np.zeros((num_points, 1), dtype=np.int64)
    # 3d boxes for each point
    boxes_3d = np.zeros((num_points, 1, 7))
    valid_boxes = np.zeros((num_points, 1, 1), dtype=np.float32)
    # add label for each object
    for label in labels:
      obj_cls_string = label['name']
      obj_cls = label_map.get(obj_cls_string, 7)
      if obj_cls >= 1 and obj_cls <= 6:
        mask = self.sel_xyz_in_box3d(label, xyz, expend_factor)
        yaw = label['yaw']
        while yaw < -0.25 * np.pi:
          yaw += np.pi
        while yaw > 0.75 * np.pi:
          yaw -= np.pi
        if yaw < 0.25 * np.pi:
          # horizontal
          cls_labels[mask, :] = obj_cls
          boxes_3d[mask, 0, :] = (label['x3d'], label['y3d'],
              label['z3d'], label['length'], label['height'],
              label['width'], yaw)
          valid_boxes[mask, 0, :] = 1
        else:
          # vertical
          cls_labels[mask, :] = obj_cls + 1
          boxes_3d[mask, 0, :] = (label['x3d'], label['y3d'],
              label['z3d'], label['length'], label['height'],
              label['width'], yaw)
          valid_boxes[mask, 0, :] = 1
      else:
        if obj_cls_string != 'DontCare':
          mask = self.sel_xyz_in_box3d(label, xyz, expend_factor)
          cls_labels[mask, :] = obj_cls
          valid_boxes[mask, 0, :] = 0.0
    return cls_labels, boxes_3d, valid_boxes, label_map

  def assign_classaware_car_label_to_points(self, labels, xyz, expend_factor):
    """Assign class label and bounding boxes to xyz points. """
    assert self.num_classes == 4
    num_points = xyz.shape[0]
    assert num_points > 0, "No point No prediction"
    assert xyz.shape[1] == 3
    # define label map
    label_map = {
        'Background': 0,
        'Car': 1,
        'DontCare': 3
        }
    # by default, all points are assigned with background label 0.
    cls_labels = np.zeros((num_points, 1), dtype=np.int64)
    # 3d boxes for each point
    boxes_3d = np.zeros((num_points, 1, 7))
    valid_boxes = np.zeros((num_points, 1, 1), dtype=np.float32)
    # add label for each object
    for label in labels:
      obj_cls_string = label['name']
      obj_cls = label_map.get(obj_cls_string, 3)
      if obj_cls >= 1 and obj_cls <= 2:
        mask = self.sel_xyz_in_box3d(label, xyz, expend_factor)
        yaw = label['yaw']
        while yaw < -0.25 * np.pi:
          yaw += np.pi
        while yaw > 0.75 * np.pi:
          yaw -= np.pi
        if yaw < 0.25 * np.pi:
          # horizontal
          cls_labels[mask, :] = obj_cls
          boxes_3d[mask, 0, :] = (label['x3d'], label['y3d'],
              label['z3d'], label['length'], label['height'],
              label['width'], yaw)
          valid_boxes[mask, 0, :] = 1
        else:
          # vertical
          cls_labels[mask, :] = obj_cls + 1
          boxes_3d[mask, 0, :] = (label['x3d'], label['y3d'],
              label['z3d'], label['length'], label['height'],
              label['width'], yaw)
          valid_boxes[mask, 0, :] = 1
      else:
        if obj_cls_string != 'DontCare':
          mask = self.sel_xyz_in_box3d(label, xyz, expend_factor)
          cls_labels[mask, :] = obj_cls
          valid_boxes[mask, 0, :] = 0.0
    return cls_labels, boxes_3d, valid_boxes, label_map

  def assign_classaware_ped_and_cyc_label_to_points(self, labels, xyz,
      expend_factor):
    """Assign class label and bounding boxes to xyz points. """
    assert self.num_classes == 6
    num_points = xyz.shape[0]
    assert num_points > 0, "No point No prediction"
    assert xyz.shape[1] == 3
    # define label map
    label_map = {
        'Background': 0,
        'Pedestrian': 1,
        'Cyclist':3,
        'DontCare': 5
        }
    # by default, all points are assigned with background label 0.
    cls_labels = np.zeros((num_points, 1), dtype=np.int64)
    # 3d boxes for each point
    boxes_3d = np.zeros((num_points, 1, 7))
    valid_boxes = np.zeros((num_points, 1, 1), dtype=np.float32)
    # add label for each object
    for label in labels:
      obj_cls_string = label['name']
      obj_cls = label_map.get(obj_cls_string, 5)
      if obj_cls >= 1 and obj_cls <= 4:
        mask = self.sel_xyz_in_box3d(label, xyz, expend_factor)
        yaw = label['yaw']
        while yaw < -0.25*np.pi:
          yaw += np.pi
        while yaw > 0.75*np.pi:
          yaw -= np.pi
        if yaw < 0.25*np.pi:
          # horizontal
          cls_labels[mask, :] = obj_cls
          boxes_3d[mask, 0, :] = (label['x3d'], label['y3d'],
              label['z3d'], label['length'], label['height'],
              label['width'], yaw)
          valid_boxes[mask, 0, :] = 1
        else:
          # vertical
          cls_labels[mask, :] = obj_cls+1
          boxes_3d[mask, 0, :] = (label['x3d'], label['y3d'],
              label['z3d'], label['length'], label['height'],
              label['width'], yaw)
          valid_boxes[mask, 0, :] = 1
      else:
        if obj_cls_string != 'DontCare':
          mask = self.sel_xyz_in_box3d(label, xyz, expend_factor)
          cls_labels[mask, :] = obj_cls
          valid_boxes[mask, 0, :] = 0.0
      return cls_labels, boxes_3d, valid_boxes, label_map

  def fetch_data(self, idx):
    def _fetch_(idx):
      cam_rgb_points = self._get_cam_points_in_image_with_rgb(idx)
      if self.is_training:
        box_label_lst = self._get_label(idx)
        cam_rgb_points, box_label_lst = self.aug_fn(cam_rgb_points, box_label_lst)
      graph_generate_fn = get_graph_generate_fn(self.graph_gen_method)
      (vertex_coord_lst, keypoint_indices_lst, edges_lst) = graph_generate_fn(cam_rgb_points.xyz, **self.config["graph_gen_kwargs"])
      if self.config['input_features'] == 'irgb':
        input_v = cam_rgb_points.attr
      elif self.config['input_features'] == '0rgb':
        input_v = np.hstack([np.zeros((cam_rgb_points.attr.shape[0], 1)),
          cam_rgb_points.attr[:, 1:]])
      elif self.config['input_features'] == '0000':
        input_v = np.zeros_like(cam_rgb_points.attr)
      elif self.config['input_features'] == 'i000':
        input_v = np.hstack([cam_rgb_points.attr[:, [0]],
            np.zeros((cam_rgb_points.attr.shape[0], 3))])
      elif self.config['input_features'] == 'i':
        input_v = cam_rgb_points.attr[:, [0]]
      elif self.config['input_features'] == '0':
        input_v = np.zeros((cam_rgb_points.attr.shape[0], 1))

      if not self.is_training:
        if self.config['label_method'] == 'yaw':
          label_map = {'Background': 0, 'Car': 1, 'Pedestrian': 3,
              'Cyclist': 5,'DontCare': 7}
        if self.config['label_method'] == 'Car':
          label_map = {'Background': 0, 'Car': 1, 'DontCare': 3}
        if self.config['label_method'] == 'Pedestrian_and_Cyclist':
          label_map = {'Background': 0, 'Pedestrian': 1, 'Cyclist':3,
              'DontCare': 5}
      else:
        last_layer_graph_level = self.config['model_kwargs'][
          'layer_configs'][-1]['graph_level']
        last_layer_points_xyz = vertex_coord_lst[last_layer_graph_level+1]
        if self.config['label_method'] == 'yaw':
            (cls_labels, boxes_3d, valid_boxes, label_map) =\
                self.assign_classaware_label_to_points(box_label_lst,
                    last_layer_points_xyz, expend_factor=(1.0, 1.0, 1.0))
        if self.config['label_method'] == 'Car':
            cls_labels, boxes_3d, valid_boxes, label_map =\
                self.assign_classaware_car_label_to_points(box_label_lst,
                    last_layer_points_xyz,  expend_factor=(1.0, 1.0, 1.0))
        if self.config['label_method'] == 'Pedestrian_and_Cyclist':
            cls_labels, boxes_3d, valid_boxes, label_map =\
                self.assign_classaware_ped_and_cyc_label_to_points(
                    box_label_lst,
                    last_layer_points_xyz,  expend_factor=(1.0, 1.0, 1.0))
        enc_boxes = self.box_encoding_fnc(cls_labels, last_layer_points_xyz, boxes_3d, label_map)
      input_v = input_v.astype(np.float32)
      vertex_coord_list = [p.astype(np.float32) for p in vertex_coord_lst]
      keypoint_indices_lst = [e.astype(np.int64) for e in keypoint_indices_lst]
      edges_list = [e.astype(np.int64) for e in edges_lst]
      if self.is_training:
        cls_labels = cls_labels.astype(np.int32)
        enc_boxes = enc_boxes.astype(np.float32)
        valid_boxes = valid_boxes.astype(np.float32)
      else:
        cls_labels, enc_boxes, valid_boxes = None, None, None
      return (input_v, vertex_coord_list, keypoint_indices_lst, edges_list,
          cls_labels, enc_boxes, valid_boxes)
    return self.mem_cache.find(idx, _fetch_)
  
  def __len__(self):
    return len(self._file_list)

class KittiPyG(Dataset):
  def __init__(self, ds_config):
    self.kitti = KittiHelper(ds_config)
  
  def get(self, idx):
    input_v, coord_lst, keypoint_indices_lst, edges, cls_labels, enc_boxes, valid_boxes = \
      self.kitti.fetch_data(idx)
    # input_v: N * d
    # coord_lst: N * 3
    # keypoints: K
    # edges: E * 2
    # we need to convert it to Data obj
    e_lens = torch.tensor([e.shape[0] for e in edges], dtype=torch.long) # TODO: no need to use tensors
    c_lens = torch.tensor([c.shape[0] for c in coord_lst], dtype=torch.long)
    k_lens = torch.tensor([k.shape[0] for k in keypoint_indices_lst], dtype=torch.long)
    edges = torch.tensor(np.concatenate(edges), dtype=torch.long).t().contiguous()
    coords = torch.tensor(np.concatenate(coord_lst), dtype=torch.float32)
    keypoints = torch.tensor(np.concatenate(keypoint_indices_lst), dtype=torch.long)
    if self.kitti.is_training:
      # TODO: labels... should be tensors
      data = Data(x=torch.Tensor(input_v), edge_index=edges, pos=coords, keypoints=keypoints, \
        labels=cls_labels, enc_boxes=enc_boxes, valid_boxes=valid_boxes, e_lens=e_lens, c_lens=c_lens, k_lens=k_lens)
    else:
      data = Data(x=torch.tensor(input_v, dtype=torch.float32), edge_index=edges, pos=coords, keypoints=keypoints, \
        e_lens=e_lens, c_lens=c_lens, k_lens=k_lens)
    return data
  
  def __len__(self):
    return len(self.kitti)

class KittiExt(Dataset):
  def __init__(self, ds_config):
    self.kitti = KittiHelper(ds_config)
  
  def __len__(self):
    return len(self.kitti)
  
  def get(self, idx):
    return self.__getitem__(idx)
  
  def __getitem__(self, idx):
    input_v, coord_lst, keypoint_indices_lst, edges_lst, cls_labels, enc_boxes, valid_boxes = \
      self.kitti.fetch_data(idx)
    graphs = []
    for i in range(len(edges_lst)):
      edges = edges_lst[i].transpose(1, 0)
      edge_vals = [1] * len(edges[0, :])
      keys = keypoint_indices_lst[i]
      coords = coord_lst[i]
      num_nodes = len(coords)
      coo = coo_matrix((edge_vals, (edges[0, :], edges[1, :])), shape=(num_nodes, num_nodes))
      csc = coo.tocsc()
      row_pointers = torch.IntTensor(csc.indptr)
      column_index = torch.IntTensor(csc.indices)
      degrees = (row_pointers[1:] - row_pointers[:-1]).tolist()
      degrees = torch.sqrt(torch.FloatTensor(list(map(lambda x: 1 if x > 0 else 0, degrees))))
      part_size = 32
      part_ptr, part_to_nodes = gnn_ext.build_part(part_size, row_pointers)
      sg = [torch.from_numpy(input_v) if i == 0 else None, torch.LongTensor(edges), \
        torch.LongTensor(keys), torch.FloatTensor(coords), row_pointers, column_index, degrees, part_ptr, part_to_nodes]
      graphs.append(sg)
    return graphs

class KittiDGL(DGLDataset):
  def __init__(self, ds_config):
    self.kitti = KittiHelper(ds_config)
  
  def __len__(self):
    return len(self.kitti)
  
  def get(self, idx):
    return self.__getitem__(idx)
  
  def __getitem__(self, idx):
    """
    there are several graphs, we have two ways to implement this:
     1. create a single graph, and cut a edge_subgraph at runtime during forward, this is very time-consuming
     2. we create subgraphs here, note that, we shouldn't set the ndata here, and should set it at runtime
    """
    input_v, coord_lst, keypoint_indices_lst, edges_lst, cls_labels, enc_boxes, valid_boxes = \
      self.kitti.fetch_data(idx)
    # LOGF(len(edges_lst), edges_lst[1], np.array(input_v).shape, keypoint_indices_lst[1].shape)
    graphs = []
    for i in range(len(edges_lst)):
      edges = edges_lst[i]
      graph = dgl.graph((torch.from_numpy(edges[:, 0]), torch.from_numpy(edges[:, 1])))
      graph.ndata["p"] = torch.from_numpy(coord_lst[i]) # N * 3
      graph.key = torch.from_numpy(keypoint_indices_lst[i])
      if i == 0:
        graph.ndata["x"] = torch.from_numpy(input_v)
      graphs.append(graph)
      # graph.e_lens = torch.tensor([e.shape[0] for e in edges_lst], dtype=torch.long)
      # graph.c_lens = torch.tensor([c.shape[0] for c in coord_lst], dtype=torch.long)
      # graph.k_lens = torch.tensor([k.shape[0] for k in keypoint_indices_lst], dtype=torch.long)
    return graphs
