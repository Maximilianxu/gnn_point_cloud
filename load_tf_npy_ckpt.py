import torch
import numpy as np
from model import MultiLayerFastLocalGraph
from omegaconf import OmegaConf
import os
from util import *

model_cfg = OmegaConf.to_container(OmegaConf.load("conf/model.yaml"))

model = MultiLayerFastLocalGraph(model_cfg)

params = list(model.parameters())
LOGF(len(params))
ckpt_path = "checkpoints/tf_npy"
ckpt_file_list = sorted(os.listdir(ckpt_path), key=lambda k: int(k.split("_")[0]))
for i, f in enumerate(ckpt_file_list):
  ckpt_file = os.path.join(ckpt_path, f)
  weight = np.load(ckpt_file)
  LOGD(i, weight.shape, list(range(len(weight.shape) - 1, -1, -1)))
  params[i].data = torch.from_numpy(weight).permute(*list(range(len(weight.shape) - 1, -1, -1)))

torch.save(model.state_dict(), "checkpoints/car_point_cloud.pth")
