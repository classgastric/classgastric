import numpy as np
import torch
import torch.nn as nn


def generate_grid(imgshape):
    x = np.arange(imgshape[0])
    y = np.arange(imgshape[1])
    grid = np.array(np.meshgrid(y, x))
    grid = torch.from_numpy(grid)
    return grid


class deform_SpatialTransform(nn.Module):
    def __init__(self):
        super(deform_SpatialTransform, self).__init__()
    def forward(self, img, flow, sample_grid):
        flow = flow.permute(0, 2, 3, 1)
        sample_grid = sample_grid.repeat(img.shape[0], 1, 1, 1).permute(0, 2, 3, 1)
        sample_grid = sample_grid + flow
        size_tensor = sample_grid.size()
        sample_grid[:, :, :, 0] = (sample_grid[:, :, :, 0] - ((size_tensor[2]-1) / 2)) / (size_tensor[2]-1) * 2
        sample_grid[:, :, :, 1] = (sample_grid[:, :, :, 1] - ((size_tensor[1]-1) / 2)) / (size_tensor[1]-1) * 2
        pred = torch.nn.functional.grid_sample(img, sample_grid, mode='bilinear')
        return pred