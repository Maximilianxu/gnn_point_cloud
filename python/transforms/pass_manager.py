import torch
import torch.nn as nn
from pass_util import *

class Pass:
  def __init__(self, name, deps=[]) -> None:
    self.name = name
    self.deps = deps

  def run(self, module):
    pass

# TODO: put this in a separate file, if neccessary
class SimpleLinearFusionPass(Pass):
  def __init__(self, deps=[], fuse_range=None) -> None:
    name = "simple_linear_fusion"
    super().__init__(name, deps)
    self.fuse_range = fuse_range
    self.deps = deps
  
  def run(self, module):
    assert isinstance(module, nn.Sequential)
    mods = flatten(module)
    assert len(mods) == flatten_all(module) # a sequential network with only one branch
    linear_ranges = []
    start, end = -1, -1
    for idx, mod in enumerate(mods):
      if isinstance(mod, nn.Linear):
        if start == -1:
          start = idx
        end = idx
      else:
        if end > 0:
          if end > start:
            linear_ranges.append([start, end])
          start, end = -1, -1
    linear_replaces = {}
    for lr in linear_ranges:
      if start == end:
        continue
      start, end = lr
      to_fuse_layers = mods[start: end + 1]
      weight_ = to_fuse_layers[0].weight
      bias_ = to_fuse_layers[0].bias
      input_feats_len = to_fuse_layers[0].in_features
      if bias_ is None:
        bias_ = torch.zeros([weight_.size(1), 1], dtype=torch.float32)
      for layer in to_fuse_layers[1:]:
        weight_ = torch.matmul(weight_, layer.weight)
        bias_ = torch.matmul(bias_, layer.weight)
        if layer.bias is not None:
          bias_ += layer.bias
      repl = nn.Linear(input_feats_len, bias_.size(0))
      repl.weight = nn.parameter(weight_)
      repl.bias = nn.parameter(bias_)
      linear_replaces[(start, end)] = repl

    ofst = 0
    for lr in linear_replaces:
      s, e = lr
      s, e = s - ofst, e - ofst
      assert s >= 0 and e >= 0
      mods[s: e + 1] = linear_replaces[lr]
      ofst += (e - s)
    return nn.Sequential(*mods)

class PassManager:
  def __init__(self) -> None:
    self.passes = []

  def register(self, opass):
    self.passes.append(opass)
  
  def transform(self, module):
    while len(self.passes) > 0:
      exec_idx = -1
      for idx, pa in enumerate(self.passes):
        deps = list(filter(lambda x: x.name in pa.deps, self.passes))
        if len(deps) > 0:
          continue
        pa.run(module)
        exec_idx = idx
        break
      self.passes.pop(exec_idx)