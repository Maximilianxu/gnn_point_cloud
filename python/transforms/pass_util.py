import torch

def flatten(model: torch.nn.Module):
  return [module for module in model.modules() if not isinstance(module, torch.nn.Sequential)]

# recursively flatten
def flatten_all(model: torch.nn.Module):
  # get children form model!
  children = list(model.children())
  flatt_children = []
  if children == []:
    # if model has no children; model is last child! :O
    return model
  else:
    # look for children from children... to the last child!
    for child in children:
      flatt_children.extend(flatten_all(child))
        
  return flatt_children