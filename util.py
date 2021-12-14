import numpy as np
from termcolor import colored
from omegaconf import OmegaConf

backend_framework = OmegaConf.load("conf/settings.yaml")["backend"] # used to set backend

def LOGF(*args):
  print(colored('=>F:', 'red', attrs=['bold']), ' '.join(list(map(lambda x: str(x), list(args)))))
  print('exiting...')
  exit(0)

def LOGI(*args):
  print(colored('=>I:', 'blue', attrs=['bold']), ' '.join(list(map(lambda x: str(x), list(args)))))

def LOGD(*args):
  print(colored('=>D:', 'yellow', attrs=['bold']), ' '.join(list(map(lambda x: str(x), list(args)))))

def load_config(path):
  return OmegaConf.to_container(OmegaConf.load(path))

class MemCache:
  def __init__(self, capacity, ways=4) -> None:
    self._capacity = capacity
    self._cache_ways = ways # e.g. 4-way set association
    self._mem_cache = [None] * self._capacity # idx: frame
    self._mem_tlb = [-1] * self._capacity # idx: frame_idx
    self._cache_timeit = np.array([0] * self._capacity) # idx: counter
    self._mem_cache = [None] * capacity
  
  def find(self, key, fetch_data_callback=None):
    idx = hash(key) if type(key) != int else key
    if idx in self._mem_tlb:
      absol_idx = self._mem_tlb.index(idx)
      self._cache_timeit[absol_idx] += 1
      return self._mem_cache[absol_idx]
    if fetch_data_callback is not None:
      if type(idx) == int:
        ret_data = fetch_data_callback(idx)
        self.write(idx, ret_data)
        for i in range(1, self._cache_ways):
          data = fetch_data_callback(idx + i)
          self.write(idx + i, data)
      return ret_data
    return None
  
  def write(self, key, data):
    idx = hash(key) if type(key) != int else key
    if idx in self._mem_tlb:
      return
    set_idx = idx % (self._capacity // self._cache_ways)
    cache_set_start = set_idx * self._cache_ways
    cache_slice = slice(cache_set_start, cache_set_start + self._cache_ways)
    min_hit = np.max(self._cache_timeit[cache_slice])
    repl_idx = -1
    for intra_idx in range(self._cache_ways):
      if self._mem_cache[cache_set_start + intra_idx] == None:
        self._mem_cache[cache_set_start + intra_idx] = data
        self._cache_timeit[cache_set_start + intra_idx] = 1
        self._mem_tlb[cache_set_start + intra_idx] = idx
        repl_idx = -1 # no need to perform replace
        min_hit = -1
        break
      if self._cache_timeit[cache_set_start + intra_idx] <= min_hit:
        min_hit = self._cache_timeit[cache_set_start + intra_idx]
        repl_idx = intra_idx
    
    # replace the min hit
    if repl_idx != - 1:
      self._mem_cache[cache_set_start + repl_idx] = data
      self._cache_timeit[cache_set_start + repl_idx] = 1
      self._mem_tlb[cache_set_start + repl_idx] = idx

# some util functions
# box enc/dec methods
def direct_box_encoding(cls_labels, points_xyz, boxes_3d):
  return boxes_3d

def direct_box_decoding(cls_labels, points_xyz, encoded_boxes):
  return encoded_boxes

def center_box_encoding(cls_labels, points_xyz, boxes_3d):
  boxes_3d[:, 0] = boxes_3d[:, 0] - points_xyz[:, 0]
  boxes_3d[:, 1] = boxes_3d[:, 1] - points_xyz[:, 1]
  boxes_3d[:, 2] = boxes_3d[:, 2] - points_xyz[:, 2]
  return boxes_3d

def center_box_decoding(cls_labels, points_xyz, encoded_boxes):
  encoded_boxes[:, 0] = encoded_boxes[:, 0] + points_xyz[:, 0]
  encoded_boxes[:, 1] = encoded_boxes[:, 1] + points_xyz[:, 1]
  encoded_boxes[:, 2] = encoded_boxes[:, 2] + points_xyz[:, 2]
  return encoded_boxes

def voxelnet_box_encoding(cls_labels, points_xyz, boxes_3d):
  # offset
  boxes_3d[:, 0] = boxes_3d[:, 0] - points_xyz[:, 0]
  boxes_3d[:, 1] = boxes_3d[:, 1] - points_xyz[:, 1]
  boxes_3d[:, 2] = boxes_3d[:, 2] - points_xyz[:, 2]
  # Car
  mask = cls_labels[:, 0] == 2
  boxes_3d[mask, 0] = boxes_3d[mask, 0]/3.9
  boxes_3d[mask, 1] = boxes_3d[mask, 1]/1.56
  boxes_3d[mask, 2] = boxes_3d[mask, 2]/1.6
  boxes_3d[mask, 3] = np.log(boxes_3d[mask, 3]/3.9)
  boxes_3d[mask, 4] = np.log(boxes_3d[mask, 4]/1.56)
  boxes_3d[mask, 5] = np.log(boxes_3d[mask, 5]/1.6)
  # Pedestrian and Cyclist
  mask = (cls_labels[:, 0] == 1) + (cls_labels[:, 0] == 3)
  boxes_3d[mask, 0] = boxes_3d[mask, 0]/0.8
  boxes_3d[mask, 1] = boxes_3d[mask, 1]/1.73
  boxes_3d[mask, 2] = boxes_3d[mask, 2]/0.6
  boxes_3d[mask, 3] = np.log(boxes_3d[mask, 3]/0.8)
  boxes_3d[mask, 4] = np.log(boxes_3d[mask, 4]/1.73)
  boxes_3d[mask, 5] = np.log(boxes_3d[mask, 5]/0.6)
  # normalize all yaws
  boxes_3d[:, 6] = boxes_3d[:, 6]/(np.pi*0.5)
  return boxes_3d

def voxelnet_box_decoding(cls_labels, points_xyz, encoded_boxes):
  # Car
  mask = cls_labels[:, 0] == 2
  encoded_boxes[mask, 0] = encoded_boxes[mask, 0]*3.9
  encoded_boxes[mask, 1] = encoded_boxes[mask, 1]*1.56
  encoded_boxes[mask, 2] = encoded_boxes[mask, 2]*1.6
  encoded_boxes[mask, 3] = np.exp(encoded_boxes[mask, 3])*3.9
  encoded_boxes[mask, 4] = np.exp(encoded_boxes[mask, 4])*1.56
  encoded_boxes[mask, 5] = np.exp(encoded_boxes[mask, 5])*1.6
  # Pedestrian and Cyclist
  mask = (cls_labels[:, 0] == 1) + (cls_labels[:, 0] == 3)
  encoded_boxes[mask, 0] = encoded_boxes[mask, 0]*0.8
  encoded_boxes[mask, 1] = encoded_boxes[mask, 1]*1.73
  encoded_boxes[mask, 2] = encoded_boxes[mask, 2]*0.6
  encoded_boxes[mask, 3] = np.exp(encoded_boxes[mask, 3])*0.8
  encoded_boxes[mask, 4] = np.exp(encoded_boxes[mask, 4])*1.73
  encoded_boxes[mask, 5] = np.exp(encoded_boxes[mask, 5])*0.6
  # offset
  encoded_boxes[:, 0] = encoded_boxes[:, 0] + points_xyz[:, 0]
  encoded_boxes[:, 1] = encoded_boxes[:, 1] + points_xyz[:, 1]
  encoded_boxes[:, 2] = encoded_boxes[:, 2] + points_xyz[:, 2]
  # recover all yaws
  encoded_boxes[:, 6] = encoded_boxes[:, 6]*(np.pi*0.5)
  return encoded_boxes

def classaware_voxelnet_box_encoding(cls_labels, points_xyz, boxes_3d):
  """
  Args:
      boxes_3d: [None, num_classes, 7]
  """
  encoded_boxes_3d = np.zeros_like(boxes_3d)
  num_classes = boxes_3d.shape[1]
  points_xyz = np.expand_dims(points_xyz, axis=1)
  points_xyz = np.tile(points_xyz, (1, num_classes, 1))
  encoded_boxes_3d[:, :, 0] = boxes_3d[:, :, 0] - points_xyz[:, :, 0]
  encoded_boxes_3d[:, :, 1] = boxes_3d[:, :, 1] - points_xyz[:, :, 1]
  encoded_boxes_3d[:, :, 2] = boxes_3d[:, :, 2] - points_xyz[:, :, 2]
  # Car horizontal
  mask = cls_labels[:, 0] == 1
  encoded_boxes_3d[mask, 0, 0] = encoded_boxes_3d[mask, 0, 0]/3.9
  encoded_boxes_3d[mask, 0, 1] = encoded_boxes_3d[mask, 0, 1]/1.56
  encoded_boxes_3d[mask, 0, 2] = encoded_boxes_3d[mask, 0, 2]/1.6
  encoded_boxes_3d[mask, 0, 3] = np.log(boxes_3d[mask, 0, 3]/3.9)
  encoded_boxes_3d[mask, 0, 4] = np.log(boxes_3d[mask, 0, 4]/1.56)
  encoded_boxes_3d[mask, 0, 5] = np.log(boxes_3d[mask, 0, 5]/1.6)
  encoded_boxes_3d[mask, 0, 6] = boxes_3d[mask, 0, 6]/(np.pi*0.25)
  # Car vertical
  mask = cls_labels[:, 0] == 2
  encoded_boxes_3d[mask, 0, 0] = encoded_boxes_3d[mask, 0, 0]/3.9
  encoded_boxes_3d[mask, 0, 1] = encoded_boxes_3d[mask, 0, 1]/1.56
  encoded_boxes_3d[mask, 0, 2] = encoded_boxes_3d[mask, 0, 2]/1.6
  encoded_boxes_3d[mask, 0, 3] = np.log(boxes_3d[mask, 0, 3]/3.9)
  encoded_boxes_3d[mask, 0, 4] = np.log(boxes_3d[mask, 0, 4]/1.56)
  encoded_boxes_3d[mask, 0, 5] = np.log(boxes_3d[mask, 0, 5]/1.6)
  encoded_boxes_3d[mask, 0, 6] = (boxes_3d[mask, 0, 6]-np.pi*0.5)/(np.pi*0.25)
  # Pedestrian horizontal
  mask = cls_labels[:, 0] == 3
  encoded_boxes_3d[mask, 0, 0] = encoded_boxes_3d[mask, 0, 0]/0.8
  encoded_boxes_3d[mask, 0, 1] = encoded_boxes_3d[mask, 0, 1]/1.73
  encoded_boxes_3d[mask, 0, 2] = encoded_boxes_3d[mask, 0, 2]/0.6
  encoded_boxes_3d[mask, 0, 3] = np.log(boxes_3d[mask, 0, 3]/0.8)
  encoded_boxes_3d[mask, 0, 4] = np.log(boxes_3d[mask, 0, 4]/1.73)
  encoded_boxes_3d[mask, 0, 5] = np.log(boxes_3d[mask, 0, 5]/0.6)
  encoded_boxes_3d[mask, 0, 6] = boxes_3d[mask, 0, 6]/(np.pi*0.25)
  # Pedestrian vertical
  mask = cls_labels[:, 0] == 4
  encoded_boxes_3d[mask, 0, 0] = encoded_boxes_3d[mask, 0, 0]/0.8
  encoded_boxes_3d[mask, 0, 1] = encoded_boxes_3d[mask, 0, 1]/1.73
  encoded_boxes_3d[mask, 0, 2] = encoded_boxes_3d[mask, 0, 2]/0.6
  encoded_boxes_3d[mask, 0, 3] = np.log(boxes_3d[mask, 0, 3]/0.8)
  encoded_boxes_3d[mask, 0, 4] = np.log(boxes_3d[mask, 0, 4]/1.73)
  encoded_boxes_3d[mask, 0, 5] = np.log(boxes_3d[mask, 0, 5]/0.6)
  encoded_boxes_3d[mask, 0, 6] = (boxes_3d[mask, 0, 6]-np.pi*0.5)/(np.pi*0.25)
  # Cyclist horizontal
  mask = cls_labels[:, 0] == 5
  encoded_boxes_3d[mask, 0, 0] = encoded_boxes_3d[mask, 0, 0]/1.76
  encoded_boxes_3d[mask, 0, 1] = encoded_boxes_3d[mask, 0, 1]/1.73
  encoded_boxes_3d[mask, 0, 2] = encoded_boxes_3d[mask, 0, 2]/0.6
  encoded_boxes_3d[mask, 0, 3] = np.log(boxes_3d[mask, 0, 3]/1.76)
  encoded_boxes_3d[mask, 0, 4] = np.log(boxes_3d[mask, 0, 4]/1.73)
  encoded_boxes_3d[mask, 0, 5] = np.log(boxes_3d[mask, 0, 5]/0.6)
  encoded_boxes_3d[mask, 0, 6] = boxes_3d[mask, 0, 6]/(np.pi*0.25)
  # Cyclist vertical
  mask = cls_labels[:, 0] == 6
  encoded_boxes_3d[mask, 0, 0] = encoded_boxes_3d[mask, 0, 0]/1.76
  encoded_boxes_3d[mask, 0, 1] = encoded_boxes_3d[mask, 0, 1]/1.73
  encoded_boxes_3d[mask, 0, 2] = encoded_boxes_3d[mask, 0, 2]/0.6
  encoded_boxes_3d[mask, 0, 3] = np.log(boxes_3d[mask, 0, 3]/1.76)
  encoded_boxes_3d[mask, 0, 4] = np.log(boxes_3d[mask, 0, 4]/1.73)
  encoded_boxes_3d[mask, 0, 5] = np.log(boxes_3d[mask, 0, 5]/0.6)
  encoded_boxes_3d[mask, 0, 6] = (boxes_3d[mask, 0, 6]-np.pi*0.5)/(np.pi*0.25)

  return encoded_boxes_3d

def classaware_voxelnet_box_decoding(cls_labels, points_xyz, encoded_boxes):
  decoded_boxes_3d = np.zeros_like(encoded_boxes)
  # Car horizontal
  mask = cls_labels[:, 0] == 1
  decoded_boxes_3d[mask, 0, 0] = encoded_boxes[mask, 0, 0]*3.9
  decoded_boxes_3d[mask, 0, 1] = encoded_boxes[mask, 0, 1]*1.56
  decoded_boxes_3d[mask, 0, 2] = encoded_boxes[mask, 0, 2]*1.6
  decoded_boxes_3d[mask, 0, 3] = np.exp(encoded_boxes[mask, 0, 3])*3.9
  decoded_boxes_3d[mask, 0, 4] = np.exp(encoded_boxes[mask, 0, 4])*1.56
  decoded_boxes_3d[mask, 0, 5] = np.exp(encoded_boxes[mask, 0, 5])*1.6
  decoded_boxes_3d[mask, 0, 6] = encoded_boxes[mask, 0, 6]*(np.pi*0.25)
  # Car vertical
  mask = cls_labels[:, 0] == 2
  decoded_boxes_3d[mask, 0, 0] = encoded_boxes[mask, 0, 0]*3.9
  decoded_boxes_3d[mask, 0, 1] = encoded_boxes[mask, 0, 1]*1.56
  decoded_boxes_3d[mask, 0, 2] = encoded_boxes[mask, 0, 2]*1.6
  decoded_boxes_3d[mask, 0, 3] = np.exp(encoded_boxes[mask, 0, 3])*3.9
  decoded_boxes_3d[mask, 0, 4] = np.exp(encoded_boxes[mask, 0, 4])*1.56
  decoded_boxes_3d[mask, 0, 5] = np.exp(encoded_boxes[mask, 0, 5])*1.6
  decoded_boxes_3d[mask, 0, 6] = (
      encoded_boxes[mask, 0, 6])*(np.pi*0.25)+0.5*np.pi
  # Pedestrian horizontal
  mask = cls_labels[:, 0] == 3
  decoded_boxes_3d[mask, 0, 0] = encoded_boxes[mask, 0, 0]*0.8
  decoded_boxes_3d[mask, 0, 1] = encoded_boxes[mask, 0, 1]*1.73
  decoded_boxes_3d[mask, 0, 2] = encoded_boxes[mask, 0, 2]*0.6
  decoded_boxes_3d[mask, 0, 3] = np.exp(encoded_boxes[mask, 0, 3])*0.8
  decoded_boxes_3d[mask, 0, 4] = np.exp(encoded_boxes[mask, 0, 4])*1.73
  decoded_boxes_3d[mask, 0, 5] = np.exp(encoded_boxes[mask, 0, 5])*0.6
  decoded_boxes_3d[mask, 0, 6] = encoded_boxes[mask, 0, 6]*(np.pi*0.25)
  # Pedestrian vertical
  mask = cls_labels[:, 0] == 4
  decoded_boxes_3d[mask, 0, 0] = encoded_boxes[mask, 0, 0]*0.8
  decoded_boxes_3d[mask, 0, 1] = encoded_boxes[mask, 0, 1]*1.73
  decoded_boxes_3d[mask, 0, 2] = encoded_boxes[mask, 0, 2]*0.6
  decoded_boxes_3d[mask, 0, 3] = np.exp(encoded_boxes[mask, 0, 3])*0.8
  decoded_boxes_3d[mask, 0, 4] = np.exp(encoded_boxes[mask, 0, 4])*1.73
  decoded_boxes_3d[mask, 0, 5] = np.exp(encoded_boxes[mask, 0, 5])*0.6
  decoded_boxes_3d[mask, 0, 6] = (
      encoded_boxes[mask, 0, 6])*(np.pi*0.25)+0.5*np.pi
  # Cyclist horizontal
  mask = cls_labels[:, 0] == 5
  decoded_boxes_3d[mask, 0, 0] = encoded_boxes[mask, 0, 0]*1.76
  decoded_boxes_3d[mask, 0, 1] = encoded_boxes[mask, 0, 1]*1.73
  decoded_boxes_3d[mask, 0, 2] = encoded_boxes[mask, 0, 2]*0.6
  decoded_boxes_3d[mask, 0, 3] = np.exp(encoded_boxes[mask, 0, 3])*1.76
  decoded_boxes_3d[mask, 0, 4] = np.exp(encoded_boxes[mask, 0, 4])*1.73
  decoded_boxes_3d[mask, 0, 5] = np.exp(encoded_boxes[mask, 0, 5])*0.6
  decoded_boxes_3d[mask, 0, 6] = encoded_boxes[mask, 0, 6]*(np.pi*0.25)
  # Cyclist vertical
  mask = cls_labels[:, 0] == 6
  decoded_boxes_3d[mask, 0, 0] = encoded_boxes[mask, 0, 0]*1.76
  decoded_boxes_3d[mask, 0, 1] = encoded_boxes[mask, 0, 1]*1.73
  decoded_boxes_3d[mask, 0, 2] = encoded_boxes[mask, 0, 2]*0.6
  decoded_boxes_3d[mask, 0, 3] = np.exp(encoded_boxes[mask, 0, 3])*1.76
  decoded_boxes_3d[mask, 0, 4] = np.exp(encoded_boxes[mask, 0, 4])*1.73
  decoded_boxes_3d[mask, 0, 5] = np.exp(encoded_boxes[mask, 0, 5])*0.6
  decoded_boxes_3d[mask, 0, 6] = (
      encoded_boxes[mask, 0, 6])*(np.pi*0.25)+0.5*np.pi
  # offset
  num_classes = encoded_boxes.shape[1]
  points_xyz = np.expand_dims(points_xyz, axis=1)
  points_xyz = np.tile(points_xyz, (1, num_classes, 1))
  decoded_boxes_3d[:, :, 0] = decoded_boxes_3d[:, :, 0] + points_xyz[:, :, 0]
  decoded_boxes_3d[:, :, 1] = decoded_boxes_3d[:, :, 1] + points_xyz[:, :, 1]
  decoded_boxes_3d[:, :, 2] = decoded_boxes_3d[:, :, 2] + points_xyz[:, :, 2]
  return decoded_boxes_3d

median_object_size_map = {
    'Cyclist': (1.76, 1.75, 0.6),
    'Van': (4.98, 2.13, 1.88),
    'Tram': (14.66, 3.61, 2.6),
    'Car': (3.88, 1.5, 1.63),
    'Misc': (2.52, 1.65, 1.51),
    'Pedestrian': (0.88, 1.77, 0.65),
    'Truck': (10.81, 3.34, 2.63),
    'Person_sitting': (0.75, 1.26, 0.59),
    # 'DontCare': (-1.0, -1.0, -1.0)
}
# 1627 Cyclist mh=1.75; mw=0.6; ml=1.76;
# 2914 Van mh=2.13; mw=1.88; ml=4.98;
# 511 Tram mh=3.61; mw=2.6; ml=14.66;
# 28742 Car mh=1.5; mw=1.63; ml=3.88;
# 973 Misc mh=1.65; mw=1.51; ml=2.52; voxelnet
# 4487 Pedestrian mh=1.77; mw=0.65; ml=0.88;
# 1094 Truck mh=3.34; mw=2.63; ml=10.81;
# 222 Person_sitting mh=1.26; mw=0.59; ml=0.75;
# 11295 DontCare mh=-1.0; mw=-1.0; ml=-1.0;

def classaware_all_class_box_encoding(cls_labels, points_xyz, boxes_3d,
    label_map):
  encoded_boxes_3d = np.copy(boxes_3d)
  num_classes = boxes_3d.shape[1]
  points_xyz = np.expand_dims(points_xyz, axis=1)
  points_xyz = np.tile(points_xyz, (1, num_classes, 1))
  encoded_boxes_3d[:, :, 0] = boxes_3d[:, :, 0] - points_xyz[:, :, 0]
  encoded_boxes_3d[:, :, 1] = boxes_3d[:, :, 1] - points_xyz[:, :, 1]
  encoded_boxes_3d[:, :, 2] = boxes_3d[:, :, 2] - points_xyz[:, :, 2]
  for cls_name in label_map:
    if cls_name == "Background" or cls_name == "DontCare":
      continue
    cls_label = label_map[cls_name]
    l, h, w = median_object_size_map[cls_name]
    mask = cls_labels[:, 0] == cls_label
    encoded_boxes_3d[mask, 0, 0] = encoded_boxes_3d[mask, 0, 0]/l
    encoded_boxes_3d[mask, 0, 1] = encoded_boxes_3d[mask, 0, 1]/h
    encoded_boxes_3d[mask, 0, 2] = encoded_boxes_3d[mask, 0, 2]/w
    encoded_boxes_3d[mask, 0, 3] = np.log(boxes_3d[mask, 0, 3]/l)
    encoded_boxes_3d[mask, 0, 4] = np.log(boxes_3d[mask, 0, 4]/h)
    encoded_boxes_3d[mask, 0, 5] = np.log(boxes_3d[mask, 0, 5]/w)
    encoded_boxes_3d[mask, 0, 6] = boxes_3d[mask, 0, 6]/(np.pi*0.25)
    # vertical
    mask = cls_labels[:, 0] == (cls_label+1)
    encoded_boxes_3d[mask, 0, 0] = encoded_boxes_3d[mask, 0, 0]/l
    encoded_boxes_3d[mask, 0, 1] = encoded_boxes_3d[mask, 0, 1]/h
    encoded_boxes_3d[mask, 0, 2] = encoded_boxes_3d[mask, 0, 2]/w
    encoded_boxes_3d[mask, 0, 3] = np.log(boxes_3d[mask, 0, 3]/l)
    encoded_boxes_3d[mask, 0, 4] = np.log(boxes_3d[mask, 0, 4]/h)
    encoded_boxes_3d[mask, 0, 5] = np.log(boxes_3d[mask, 0, 5]/w)
    encoded_boxes_3d[mask, 0, 6] = (
        boxes_3d[mask, 0, 6]-np.pi*0.5)/(np.pi*0.25)
  return encoded_boxes_3d

def classaware_all_class_box_decoding(cls_labels, points_xyz, encoded_boxes,
    label_map):
  decoded_boxes_3d = np.copy(encoded_boxes)
  for cls_name in label_map:
    if cls_name == "Background" or cls_name == "DontCare":
      continue
    cls_label = label_map[cls_name]
    l, h, w = median_object_size_map[cls_name]
    # Car horizontal
    mask = cls_labels[:, 0] == cls_label
    decoded_boxes_3d[mask, 0, 0] = encoded_boxes[mask, 0, 0]*l
    decoded_boxes_3d[mask, 0, 1] = encoded_boxes[mask, 0, 1]*h
    decoded_boxes_3d[mask, 0, 2] = encoded_boxes[mask, 0, 2]*w
    decoded_boxes_3d[mask, 0, 3] = np.exp(encoded_boxes[mask, 0, 3])*l
    decoded_boxes_3d[mask, 0, 4] = np.exp(encoded_boxes[mask, 0, 4])*h
    decoded_boxes_3d[mask, 0, 5] = np.exp(encoded_boxes[mask, 0, 5])*w
    decoded_boxes_3d[mask, 0, 6] = encoded_boxes[mask, 0, 6]*(np.pi*0.25)
    # Car vertical
    mask = cls_labels[:, 0] == (cls_label+1)
    decoded_boxes_3d[mask, 0, 0] = encoded_boxes[mask, 0, 0]*l
    decoded_boxes_3d[mask, 0, 1] = encoded_boxes[mask, 0, 1]*h
    decoded_boxes_3d[mask, 0, 2] = encoded_boxes[mask, 0, 2]*w
    decoded_boxes_3d[mask, 0, 3] = np.exp(encoded_boxes[mask, 0, 3])*l
    decoded_boxes_3d[mask, 0, 4] = np.exp(encoded_boxes[mask, 0, 4])*h
    decoded_boxes_3d[mask, 0, 5] = np.exp(encoded_boxes[mask, 0, 5])*w
    decoded_boxes_3d[mask, 0, 6] = (
        encoded_boxes[mask, 0, 6])*(np.pi*0.25)+0.5*np.pi
  # offset
  num_classes = encoded_boxes.shape[1]
  points_xyz = np.expand_dims(points_xyz, axis=1)
  points_xyz = np.tile(points_xyz, (1, num_classes, 1))
  decoded_boxes_3d[:, :, 0] = decoded_boxes_3d[:, :, 0] + points_xyz[:, :, 0]
  decoded_boxes_3d[:, :, 1] = decoded_boxes_3d[:, :, 1] + points_xyz[:, :, 1]
  decoded_boxes_3d[:, :, 2] = decoded_boxes_3d[:, :, 2] + points_xyz[:, :, 2]
  return decoded_boxes_3d

def classaware_all_class_box_canonical_encoding(cls_labels, points_xyz,
  boxes_3d, label_map):
  boxes_3d = np.copy(boxes_3d)
  num_classes = boxes_3d.shape[1]
  points_xyz = np.expand_dims(points_xyz, axis=1)
  points_xyz = np.tile(points_xyz, (1, num_classes, 1))
  boxes_3d[:, :, 0] = boxes_3d[:, :, 0] - points_xyz[:, :, 0]
  boxes_3d[:, :, 1] = boxes_3d[:, :, 1] - points_xyz[:, :, 1]
  boxes_3d[:, :, 2] = boxes_3d[:, :, 2] - points_xyz[:, :, 2]
  encoded_boxes_3d = np.copy(boxes_3d)
  for cls_name in label_map:
    if cls_name == "Background" or cls_name == "DontCare":
      continue

    cls_label = label_map[cls_name]
    l, h, w = median_object_size_map[cls_name]
    mask = cls_labels[:, 0] == cls_label
    encoded_boxes_3d[mask, 0, 0] = (
        boxes_3d[mask, 0, 0]*np.cos(boxes_3d[mask, 0, 6]) \
        -boxes_3d[mask, 0, 2]*np.sin(boxes_3d[mask, 0, 6]))/l
    encoded_boxes_3d[mask, 0, 1] = boxes_3d[mask, 0, 1]/h
    encoded_boxes_3d[mask, 0, 2] = (
        boxes_3d[mask, 0, 0]*np.sin(boxes_3d[mask, 0, 6]) \
        +boxes_3d[mask, 0, 2]*np.cos(boxes_3d[mask, 0, 6]))/w
    encoded_boxes_3d[mask, 0, 3] = np.log(boxes_3d[mask, 0, 3]/l)
    encoded_boxes_3d[mask, 0, 4] = np.log(boxes_3d[mask, 0, 4]/h)
    encoded_boxes_3d[mask, 0, 5] = np.log(boxes_3d[mask, 0, 5]/w)
    encoded_boxes_3d[mask, 0, 6] = boxes_3d[mask, 0, 6]/(np.pi*0.25)
    # vertical
    mask = cls_labels[:, 0] == (cls_label+1)
    encoded_boxes_3d[mask, 0, 0] = (
        boxes_3d[mask, 0, 0]*np.cos(boxes_3d[mask, 0, 6]-np.pi*0.5) \
        -boxes_3d[mask, 0, 2]*np.sin(boxes_3d[mask, 0, 6]-np.pi*0.5))/w
    encoded_boxes_3d[mask, 0, 1] = boxes_3d[mask, 0, 1]/h
    encoded_boxes_3d[mask, 0, 2] = (
        boxes_3d[mask, 0, 0]*np.sin(boxes_3d[mask, 0, 6]-np.pi*0.5) \
        +boxes_3d[mask, 0, 2]*np.cos(boxes_3d[mask, 0, 6]-np.pi*0.5))/l
    encoded_boxes_3d[mask, 0, 3] = np.log(boxes_3d[mask, 0, 3]/l)
    encoded_boxes_3d[mask, 0, 4] = np.log(boxes_3d[mask, 0, 4]/h)
    encoded_boxes_3d[mask, 0, 5] = np.log(boxes_3d[mask, 0, 5]/w)
    encoded_boxes_3d[mask, 0, 6] = (
        boxes_3d[mask, 0, 6]-np.pi*0.5)/(np.pi*0.25)
  return encoded_boxes_3d

def classaware_all_class_box_canonical_decoding(cls_labels, points_xyz,
  encoded_boxes, label_map):
  decoded_boxes_3d = np.copy(encoded_boxes)
  for cls_name in label_map:
    if cls_name == "Background" or cls_name == "DontCare":
      continue
    cls_label = label_map[cls_name]
    l, h, w = median_object_size_map[cls_name]
    # Car horizontal
    mask = cls_labels[:, 0] == cls_label

    decoded_boxes_3d[mask, 0, 0] = encoded_boxes[mask, 0, 0]*l*np.cos(
        encoded_boxes[mask, 0, 6]*(np.pi*0.25))\
        +encoded_boxes[mask, 0, 2]*w*np.sin(
        encoded_boxes[mask, 0, 6]*(np.pi*0.25))
    decoded_boxes_3d[mask, 0, 1] = encoded_boxes[mask, 0, 1]*h
    decoded_boxes_3d[mask, 0, 2] = -encoded_boxes[mask, 0, 0]*l*np.sin(
        encoded_boxes[mask, 0, 6]*(np.pi*0.25))\
        +encoded_boxes[mask, 0, 2]*w*np.cos(
        encoded_boxes[mask, 0, 6]*(np.pi*0.25))

    decoded_boxes_3d[mask, 0, 3] = np.exp(encoded_boxes[mask, 0, 3])*l
    decoded_boxes_3d[mask, 0, 4] = np.exp(encoded_boxes[mask, 0, 4])*h
    decoded_boxes_3d[mask, 0, 5] = np.exp(encoded_boxes[mask, 0, 5])*w
    decoded_boxes_3d[mask, 0, 6] = encoded_boxes[mask, 0, 6]*(np.pi*0.25)

    # Car vertical
    mask = cls_labels[:, 0] == (cls_label+1)
    decoded_boxes_3d[mask, 0, 0] = encoded_boxes[mask, 0, 0]*w*np.cos(
        encoded_boxes[mask, 0, 6]*(np.pi*0.25))\
        +encoded_boxes[mask, 0, 2]*l*np.sin(
        encoded_boxes[mask, 0, 6]*(np.pi*0.25))
    decoded_boxes_3d[mask, 0, 1] = encoded_boxes[mask, 0, 1]*h
    decoded_boxes_3d[mask, 0, 2] = -encoded_boxes[mask, 0, 0]*w*np.sin(
        encoded_boxes[mask, 0, 6]*(np.pi*0.25))\
        +encoded_boxes[mask, 0, 2]*l*np.cos(
        encoded_boxes[mask, 0, 6]*(np.pi*0.25))

    decoded_boxes_3d[mask, 0, 3] = np.exp(encoded_boxes[mask, 0, 3])*l
    decoded_boxes_3d[mask, 0, 4] = np.exp(encoded_boxes[mask, 0, 4])*h
    decoded_boxes_3d[mask, 0, 5] = np.exp(encoded_boxes[mask, 0, 5])*w
    decoded_boxes_3d[mask, 0, 6] = (
        encoded_boxes[mask, 0, 6])*(np.pi*0.25)+0.5*np.pi
  # offset
  num_classes = encoded_boxes.shape[1]
  points_xyz = np.expand_dims(points_xyz, axis=1)
  points_xyz = np.tile(points_xyz, (1, num_classes, 1))
  decoded_boxes_3d[:, :, 0] = decoded_boxes_3d[:, :, 0] + points_xyz[:, :, 0]
  decoded_boxes_3d[:, :, 1] = decoded_boxes_3d[:, :, 1] + points_xyz[:, :, 1]
  decoded_boxes_3d[:, :, 2] = decoded_boxes_3d[:, :, 2] + points_xyz[:, :, 2]
  return decoded_boxes_3d

from collections import namedtuple
Points = namedtuple('Points', ['xyz', 'attr'])

def downsample_by_average_voxel(points, voxel_size):
  """Voxel downsampling using average function.

  points: a Points namedtuple containing "xyz" and "attr".
  voxel_size: the size of voxel cells used for downsampling.
  """
  # create voxel grid
  xmax, ymax, zmax = np.amax(points.xyz, axis=0)
  xmin, ymin, zmin = np.amin(points.xyz, axis=0)
  xyz_offset = np.asarray([[xmin, ymin, zmin]])
  xyz_zeros = np.asarray([0, 0, 0], dtype=np.float32)
  xyz_idx = (points.xyz - xyz_offset) // voxel_size
  xyz_idx = xyz_idx.astype(np.int32)
  dim_x, dim_y, dim_z = np.amax(xyz_idx, axis=0) + 1
  keys = xyz_idx[:, 0] + xyz_idx[:, 1]*dim_x + xyz_idx[:, 2]*dim_y*dim_x
  order = np.argsort(keys)
  keys = keys[order]
  points_xyz = points.xyz[order]
  unique_keys, lens = np.unique(keys, return_counts=True)
  indices = np.hstack([[0], lens[:-1]]).cumsum()
  downsampled_xyz = np.add.reduceat(
      points_xyz, indices, axis=0)/lens[:,np.newaxis]
  include_attr = points.attr is not None
  if include_attr:
      attr = points.attr[order]
      downsampled_attr = np.add.reduceat(
          attr, indices, axis=0)/lens[:,np.newaxis]
  if include_attr:
    return Points(xyz=downsampled_xyz, attr=downsampled_attr)
  else:
    return Points(xyz=downsampled_xyz, attr=None)

def downsample_by_random_voxel(points, voxel_size, add_rnd3d=False):
  """Downsample the points using base_voxel_size at different scales"""
  xmax, ymax, zmax = np.amax(points.xyz, axis=0)
  xmin, ymin, zmin = np.amin(points.xyz, axis=0)
  xyz_offset = np.asarray([[xmin, ymin, zmin]])
  xyz_zeros = np.asarray([0, 0, 0], dtype=np.float32)

  if not add_rnd3d:
    xyz_idx = (points.xyz - xyz_offset) // voxel_size
  else:
    xyz_idx = (points.xyz - xyz_offset + voxel_size*np.random.random((1,3))) // voxel_size
  dim_x, dim_y, dim_z = np.amax(xyz_idx, axis=0) + 1
  keys = xyz_idx[:, 0] + xyz_idx[:, 1]*dim_x + xyz_idx[:, 2]*dim_y*dim_x
  num_points = xyz_idx.shape[0]

  voxels_idx = {}
  for pidx in range(len(points.xyz)):
    key = keys[pidx]
    if key in voxels_idx:
      voxels_idx[key].append(pidx)
    else:
      voxels_idx[key] = [pidx]

  downsampled_xyz = []
  downsampled_attr = []
  for key in voxels_idx:
    center_idx = np.random.choice(voxels_idx[key])
    downsampled_xyz.append(points.xyz[center_idx])
    downsampled_attr.append(points.attr[center_idx])

  return Points(xyz=np.array(downsampled_xyz), attr=np.array(downsampled_attr))

import open3d
from sklearn.neighbors import NearestNeighbors
# graph generations
def multi_layer_downsampling(points_xyz, base_voxel_size, levels=[1],
  add_rnd3d=False,):
  """Downsample the points using base_voxel_size at different scales"""
  xmax, ymax, zmax = np.amax(points_xyz, axis=0)
  xmin, ymin, zmin = np.amin(points_xyz, axis=0)
  xyz_offset = np.asarray([[xmin, ymin, zmin]])
  xyz_zeros = np.asarray([0, 0, 0], dtype=np.float32)
  downsampled_list = [points_xyz]
  last_level = 0
  for level in levels:
    if np.isclose(last_level, level):
      downsampled_list.append(np.copy(downsampled_list[-1]))
    else:
      if add_rnd3d:
        xyz_idx = (points_xyz-xyz_offset +
            base_voxel_size*level*np.random.random((1,3)))//\
                (base_voxel_size*level)
        xyz_idx = xyz_idx.astype(np.int32)
        dim_x, dim_y, dim_z = np.amax(xyz_idx, axis=0) + 1
        keys = xyz_idx[:, 0]+xyz_idx[:, 1]*dim_x+\
            xyz_idx[:, 2]*dim_y*dim_x
        sorted_order = np.argsort(keys)
        sorted_keys = keys[sorted_order]
        sorted_points_xyz = points_xyz[sorted_order]
        _, lens = np.unique(sorted_keys, return_counts=True)
        indices = np.hstack([[0], lens[:-1]]).cumsum()
        downsampled_xyz = np.add.reduceat(
            sorted_points_xyz, indices, axis=0)/lens[:,np.newaxis]
        downsampled_list.append(np.array(downsampled_xyz))
      else:
        pcd = open3d.PointCloud()
        pcd.points = open3d.Vector3dVector(points_xyz)
        downsampled_xyz = np.asarray(open3d.voxel_down_sample(
            pcd, voxel_size = base_voxel_size*level).points)
        downsampled_list.append(downsampled_xyz)
    last_level = level
  return downsampled_list

def multi_layer_downsampling_random(points_xyz, base_voxel_size, levels=[1],
    add_rnd3d=False):
  """Downsample the points at different scales by randomly select a point
  within a voxel cell.

  Args:
      points_xyz: a [N, D] matrix. N is the total number of the points. D is
      the dimension of the coordinates.
      base_voxel_size: scalar, the cell size of voxel.
      level_configs: a dict of 'level', 'graph_gen_method',
      'graph_gen_kwargs', 'graph_scale'.
      add_rnd3d: boolean, whether to add random offset when downsampling.

  returns: vertex_coord_list, keypoint_indices_list
  """
  xmax, ymax, zmax = np.amax(points_xyz, axis=0)
  xmin, ymin, zmin = np.amin(points_xyz, axis=0)
  xyz_offset = np.asarray([[xmin, ymin, zmin]])
  xyz_zeros = np.asarray([0, 0, 0], dtype=np.float32)
  vertex_coord_list = [points_xyz]
  keypoint_indices_list = []
  last_level = 0
  for level in levels:
    last_points_xyz = vertex_coord_list[-1]
    if np.isclose(last_level, level):
      # same downsample scale (gnn layer), just copy it
      vertex_coord_list.append(np.copy(last_points_xyz))
      keypoint_indices_list.append(
          np.expand_dims(np.arange(len(last_points_xyz)), axis=1))
    else:
      if not add_rnd3d:
        xyz_idx = (last_points_xyz - xyz_offset) \
            // (base_voxel_size*level)
      else:
        xyz_idx = (last_points_xyz - xyz_offset +
            base_voxel_size*level*np.random.random((1,3))) \
                // (base_voxel_size*level)
      xyz_idx = xyz_idx.astype(np.int32)
      dim_x, dim_y, dim_z = np.amax(xyz_idx, axis=0) + 1
      keys = xyz_idx[:, 0]+xyz_idx[:, 1]*dim_x+xyz_idx[:, 2]*dim_y*dim_x
      num_points = xyz_idx.shape[0]

      voxels_idx = {}
      for pidx in range(len(last_points_xyz)):
        key = keys[pidx]
        if key in voxels_idx:
          voxels_idx[key].append(pidx)
        else:
          voxels_idx[key] = [pidx]

      downsampled_xyz = []
      downsampled_xyz_idx = []
      for key in voxels_idx:
        center_idx = np.random.choice(voxels_idx[key])
        downsampled_xyz.append(last_points_xyz[center_idx])
        downsampled_xyz_idx.append(center_idx)
      vertex_coord_list.append(np.array(downsampled_xyz))
      keypoint_indices_list.append(
          np.expand_dims(np.array(downsampled_xyz_idx),axis=1))
    last_level = level
  return vertex_coord_list, keypoint_indices_list

def multi_layer_downsampling_select(points_xyz, base_voxel_size, levels=[1],
    add_rnd3d=False):
  """Downsample the points at different scales and match the downsampled
  points to original points by a nearest neighbor search.

  Args:
      points_xyz: a [N, D] matrix. N is the total number of the points. D is
      the dimension of the coordinates.
      base_voxel_size: scalar, the cell size of voxel.
      level_configs: a dict of 'level', 'graph_gen_method',
      'graph_gen_kwargs', 'graph_scale'.
      add_rnd3d: boolean, whether to add random offset when downsampling.

  returns: vertex_coord_list, keypoint_indices_list
  """
  # Voxel downsampling
  vertex_coord_list = multi_layer_downsampling(
      points_xyz, base_voxel_size, levels=levels, add_rnd3d=add_rnd3d)
  num_levels = len(vertex_coord_list)
  assert num_levels == len(levels) + 1
  # Match downsampled vertices to original by a nearest neighbor search.
  keypoint_indices_list = []
  last_level = 0
  for i in range(1, num_levels):
    current_level = levels[i-1]
    base_points = vertex_coord_list[i-1]
    current_points = vertex_coord_list[i]
    if np.isclose(current_level, last_level):
      # same downsample scale (gnn layer),
      # just copy it, no need to search.
      vertex_coord_list[i] = base_points
      keypoint_indices_list.append(
          np.expand_dims(np.arange(base_points.shape[0]),axis=1))
    else:
      # different scale (pooling layer), search original points.
      nbrs = NearestNeighbors(n_neighbors=1,
          algorithm='kd_tree', n_jobs=1).fit(base_points)
      indices = nbrs.kneighbors(current_points, return_distance=False)
      vertex_coord_list[i] = base_points[indices[:, 0], :]
      keypoint_indices_list.append(indices)
    last_level = current_level
  return vertex_coord_list, keypoint_indices_list

def gen_multi_level_local_graph_v3(
  points_xyz, base_voxel_size, level_configs, add_rnd3d=False,
  downsample_method='center'):
  """Generating graphs at multiple scale. This function enforce output
  vertices of a graph matches the input vertices of next graph so that
  gnn layers can be applied sequentially.

  Args:
      points_xyz: a [N, D] matrix. N is the total number of the points. D is
      the dimension of the coordinates.
      base_voxel_size: scalar, the cell size of voxel.
      level_configs: a dict of 'level', 'graph_gen_method',
      'graph_gen_kwargs', 'graph_scale'.
      add_rnd3d: boolean, whether to add random offset when downsampling.
      downsample_method: string, the name of downsampling method.
  returns: vertex_coord_list, keypoint_indices_list, edges_list
  """
  if isinstance(base_voxel_size, list):
    base_voxel_size = np.array(base_voxel_size)
  # Gather the downsample scale for each graph
  scales = [config['graph_scale'] for config in level_configs]
  # Generate vertex coordinates
  if downsample_method=='center':
    vertex_coord_list, keypoint_indices_list = \
      multi_layer_downsampling_select(
          points_xyz, base_voxel_size, scales, add_rnd3d=add_rnd3d)
  if downsample_method=='random':
    vertex_coord_list, keypoint_indices_list = \
      multi_layer_downsampling_random(
          points_xyz, base_voxel_size, scales, add_rnd3d=add_rnd3d)
  # Create edges
  edges_list = []
  for config in level_configs:
    graph_level = config['graph_level']
    gen_graph_fn = get_graph_generate_fn(config['graph_gen_method'])
    method_kwarg = config['graph_gen_kwargs']
    points_xyz = vertex_coord_list[graph_level]
    center_xyz = vertex_coord_list[graph_level+1]
    vertices = gen_graph_fn(points_xyz, center_xyz, **method_kwarg)
    edges_list.append(vertices)
  return vertex_coord_list, keypoint_indices_list, edges_list

def gen_disjointed_rnn_local_graph_v3(
    points_xyz, center_xyz, radius, num_neighbors,
    neighbors_downsample_method='random',
    scale=None):
  """Generate a local graph by radius neighbors.
  """
  # LOGD(points_xyz.reshape(-1)[:10], center_xyz.reshape(-1)[:10], radius, num_neighbors, neighbors_downsample_method, scale)
  if scale is not None:
    scale = np.array(scale)
    points_xyz = points_xyz / scale
    center_xyz = center_xyz / scale
  nbrs = NearestNeighbors(
    radius=radius,algorithm='ball_tree', n_jobs=1, ).fit(points_xyz)
  indices = nbrs.radius_neighbors(center_xyz, return_distance=False)
  if num_neighbors > 0:
    if neighbors_downsample_method == 'random':
      indices = [neighbors if neighbors.size <= num_neighbors else
              np.random.choice(neighbors, num_neighbors, replace=False)
              for neighbors in indices]
  vertices_v = np.concatenate(indices)
  vertices_i = np.concatenate(
      [i*np.ones(neighbors.size, dtype=np.int32)
          for i, neighbors in enumerate(indices)])
  vertices = np.array([vertices_v, vertices_i]).transpose()
  return vertices

# some graph gen helper functions
def get_box_encoding_len(box_enc_method):
    encoding_len_dict = {
      'direct_encoding': 7,
      'center_box_encoding': 7,
      'voxelnet_box_encoding':7,
      'classaware_voxelnet_box_encoding': 7,
      'classaware_all_class_box_encoding': 7,
      'classaware_all_class_box_canonical_encoding': 7,
    }
    return encoding_len_dict[box_enc_method]
  
def get_box_encoding_fn(box_enc_method):
  encoding_method_dict = {
    'direct_encoding': direct_box_encoding,
    'center_box_encoding': center_box_encoding,
    'voxelnet_box_encoding': voxelnet_box_encoding,
    'classaware_voxelnet_box_encoding': classaware_voxelnet_box_encoding,
    'classaware_all_class_box_encoding':classaware_all_class_box_encoding,
    'classaware_all_class_box_canonical_encoding':
        classaware_all_class_box_canonical_encoding,
  }
  return encoding_method_dict[box_enc_method]

def get_box_decoding_fn(box_enc_method):
  decoding_method_dict = {
    'direct_encoding': direct_box_decoding,
    'center_box_encoding': center_box_decoding,
    'voxelnet_box_encoding': voxelnet_box_decoding,
    'classaware_voxelnet_box_encoding': classaware_voxelnet_box_decoding,
    'classaware_all_class_box_encoding': classaware_all_class_box_decoding,
    'classaware_all_class_box_canonical_encoding':
        classaware_all_class_box_canonical_decoding,
  }
  return decoding_method_dict[box_enc_method]

def get_graph_generate_fn(method_name):
  method_map = {
      'disjointed_rnn_local_graph_v3': gen_disjointed_rnn_local_graph_v3,
      'multi_level_local_graph_v3': gen_multi_level_local_graph_v3,
  }
  return method_map[method_name]
