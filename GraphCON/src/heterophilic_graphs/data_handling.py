from torch_geometric.datasets import WebKB, WikipediaNetwork
import torch
import numpy as np

# DATA_PATH = '../../data'

def get_data(name, split=0):
  path = '../../data/'+name
  if name in ['cornell','wisconsin','texas']:
    dataset = WebKB(path,name=name)
    splits_file = np.load(f'{path}/{name}/raw/{name}_split_0.6_0.2_{split}.npz')

  elif name in ['chameleon', 'crocodile', 'squirrel']:
    dataset = WikipediaNetwork(path, name=name)
    splits_file = np.load(f'{path}/{name}/geom_gcn/raw/{name}_split_0.6_0.2_{split}.npz')
  
  data = dataset[0]
  train_mask = splits_file['train_mask']
  val_mask = splits_file['val_mask']
  test_mask = splits_file['test_mask']

  data.train_mask = torch.tensor(train_mask, dtype=torch.bool)
  data.val_mask = torch.tensor(val_mask, dtype=torch.bool)
  data.test_mask = torch.tensor(test_mask, dtype=torch.bool)

  return data
