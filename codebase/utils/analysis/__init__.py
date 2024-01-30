"""
@author: Jiahua Wu
@contact: jhwu@shu.edu.cn
"""

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import tqdm

def collect_feature(data_loader: DataLoader, feature_extractor: nn.Module,
                    device: torch.device, max_num_features:int=None)->torch.Tensor:
    """
    Fetch data from `data_loader`, and then use `feature_extracter` to collect features

    Args:
        data_loader (torch.utils.data.DataLoader): Data loader
        feature_extractor (torch.nn.Module): A feature extractor
        device (torch.device)
        max_num_features (int): The max number of features to return

    Returns:
        Features in shape (min(len(data_loader), max_num_features * mini-batch size)), :math:`|\mathcal{F}|`
    """
    feature_extractor.eval()
    all_features = []
    label = []
    with torch.no_grad():
        for index, data in enumerate(tqdm.tqdm(data_loader)):
            if max_num_features is not None and index >= max_num_features:
                break
            input = data[0].to(device)
            feature = feature_extractor(input).cpu()
            all_features.append(feature)
            label.append(data[1].cpu())
    return torch.cat(all_features, dim = 0), torch.cat(label, dim = 0)
