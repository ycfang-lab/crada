"""
@author: Jiahua Wu
@contact: jhwu@shu.edu.cn
"""

import torch
import matplotlib

matplotlib.use('Agg') #don't display directly when running
import matplotlib.pyplot as plt
import matplotlib.colors as col
import numpy as np
from sklearn.manifold import TSNE

def visualize(source_feature: torch.Tensor, target_feature: torch.Tensor,
              filename: str, source_color='r', target_color='b'):
   """
   Visualize features from different domain using t-SNE

   Args:
       source_feature (torch.Tensor): features from souce domain in shape :math:`(minibatch, F)`
       target_feature (torch.Tensor): features from target domain in shape :math:`(minibatch, F)`
       filename (str): the file name to save t-SNE
       source_color (str): the color of the source features. Default: 'r'
       target_color (str): the color of the target featrues. Default: 'b'

   Returns:

   """
   source_feature = source_feature.numpy()
   target_feature = target_feature.numpy()
   features = np.concatenate([source_feature, target_feature], axis = 0)

   # map features to 2-d using TSNE
   X_tnse = TSNE(n_components=2, random_state=33).fit_transform(features)

   # domain labels, 1 represents source while 0 represents target
   domains = np.concatenate((np.ones(len(source_feature)), np.zeros(len(target_feature))))

   #visualize using matplotlib
   fig, ax = plt.subplots(figsize=(10,10))
   ax.spines['top'].set_visible(False)
   ax.spines['right'].set_visible(False)
   ax.spines['bottom'].set_visible(False)
   ax.spines['left'].set_visible(False)
   plt.scatter(X_tnse[:, 0], X_tnse[:, 1], c = domains, cmap=col.ListedColormap([target_color, source_color])
               ,s = 20)
   plt.xticks([])
   plt.yticks([])
   plt.savefig(filename)
   
def visualize_w_category_color(source_feature: torch.Tensor, target_feature: torch.Tensor, filename: str, source_label: torch.Tensor, target_label:torch.Tensor, colors):
   """
   Visualize features from different domain using t-SNE

   Args:
       source_feature (torch.Tensor): features from souce domain in shape :math:`(minibatch, F)`
       target_feature (torch.Tensor): features from target domain in shape :math:`(minibatch, F)`
       filename (str): the file name to save t-SNE
       source_color (str): the color of the source features. Default: 'r'
       target_color (str): the color of the target featrues. Default: 'b'

   Returns:

   """
   source_feature = source_feature.numpy()
   target_feature = target_feature.numpy()
   features = np.concatenate([source_feature, target_feature], axis = 0)

   # map features to 2-d using TSNE
   X_tnse = TSNE(n_components=2, random_state=33).fit_transform(features)

   # domain labels, 1 represents source while 0 represents target
   domains = np.concatenate([source_label.numpy(), target_label.numpy()], axis = 0)
   #visualize using matplotlib
   fig, ax = plt.subplots(figsize=(10,10))
   ax.spines['top'].set_visible(False)
   ax.spines['right'].set_visible(False)
   ax.spines['bottom'].set_visible(False)
   ax.spines['left'].set_visible(False)
   plt.scatter(X_tnse[:, 0], X_tnse[:, 1], c = domains, cmap=col.ListedColormap(colors)
               ,s = 20)
   plt.xticks([])
   plt.yticks([])
   plt.savefig(filename)


