import os
import hydra
import torch
from omegaconf import OmegaConf
from dataset import KittiDGL, KittiExt, KittiPyG
from model import MultiLayerFastLocalGraph
import numpy as np
import tqdm

np.random.seed(43)
from util import *
hydra.output_subdir = None

conf_path = "/root/workspaces/python/pyg/conf/"

def test_mem_cache():
  mem_cache = MemCache(64)
  fetch_callback = lambda x: {x: str(x)}
  rslt = mem_cache.find(1, fetch_callback)
  LOGD(rslt)
  rslt = mem_cache.find(1)
  LOGD(rslt)

@hydra.main(config_path=conf_path, config_name="dataset")
def test_dataset(config):
  ds_config = OmegaConf.to_container(config)
  kids = KittiDGL(ds_config)
  LOGD("dataset len:", len(kids))
  data = kids.get(0)
  LOGD(data.c_lens, data.e_lens, data.k_lens)

# @hydra.main(config_path=conf_path, config_name="model")
def test_model():
  model_config = OmegaConf.to_container(OmegaConf.load(os.path.join(conf_path, "model.yaml")))
  dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  rest_path = "/root/workspaces/python/pyg/checkpoints/car_point_cloud.pth"

  model = MultiLayerFastLocalGraph(model_config)
  rest_state = torch.load(rest_path)
  model.load_state_dict(rest_state)
  model = model.eval()
  model = model.to(dev)

  ds_config = OmegaConf.to_container(OmegaConf.load(os.path.join(conf_path, "dataset.yaml")))
  data_helper = {"dgl": KittiDGL, "pyg": KittiPyG, "torch": KittiPyG, "ext": KittiExt}

  kids = data_helper[backend_framework](ds_config)

  from torch.profiler import profile, record_function, ProfilerActivity
  with torch.no_grad():
    for i in tqdm.trange(len(kids)):
      data = kids.get(i)
      # with profile(activities=[
      #     ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as prof:
      #   with record_function("model_inference"):
      logits, enc_boxes = model(data)
      # LOGD(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
      # prof.export_chrome_trace("pytorch_trace.json")
      logits = logits.detach().cpu()
      enc_boxes = enc_boxes.detach().cpu()

if __name__ == "__main__":
  # test_dataset()
  test_model()
  # test_mem_cache()