import torch
from . import minarearect

def minaerarect(pred):                      # pred torch.size(num_topk, points_num*2)
    rbbox = minarearect.minareabbox(pred)   # rbbox  torch.size(num_topk*8)  9points(18 channel) -> 1rbox(8 channel)
    rbbox = rbbox.reshape(-1, 8)            # rbbox torch.size(num_topk, 8)
    return rbbox