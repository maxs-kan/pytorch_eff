import numpy as np
import matplotlib
import imageio
import matplotlib.pyplot as plt
import torch
from utils import util
import os
import logging

class Visualizer():
    def __init__(self, opt):
        self.opt = opt
        self.logger = logging.getLogger()
            
    def plot_img(self, img_dict):
        A_imgs = util.tensor2im(img_dict['real_img_A'], self.opt, isDepth=False)
        A_depth = util.tensor2im(img_dict['real_depth_A'], self.opt, isDepth=True)
        A_img_depht = util.tensor2im(img_dict['img_depth_A'], self.opt, isDepth=True)
        A_norm = util.get_normals(A_depth * 1000)
        B_depth_fake = util.tensor2im(img_dict['fake_depth_B'], self.opt, isDepth=True)
        B_norm_fake = util.get_normals(B_depth_fake * 1000)

        
        B_imgs = util.tensor2im(img_dict['real_img_B'], self.opt, isDepth=False)
        B_depth = util.tensor2im(img_dict['real_depth_B'], self.opt, isDepth=True)
        B_norm = util.get_normals(B_depth * 1000)
        A_depth_fake = util.tensor2im(img_dict['fake_depth_A'], self.opt, isDepth=True)
        A_norm_fake = util.get_normals(A_depth_fake * 1000)
        B_depth_rec = util.tensor2im(img_dict['rec_depth_B'], self.opt, isDepth=True)
        B_norm_rec = util.get_normals(B_depth_rec * 1000)

        
        max_dist = self.opt.max_distance/1000
        batch_size = A_imgs.shape[0]
        n_pic = min(batch_size, self.opt.n_pic)
        n_col = 9
        fig_size = (40,30)
        n_row = 2 * n_pic
        fig, axes = plt.subplots(nrows=n_row, ncols=n_col, figsize=fig_size)
        fig.subplots_adjust(hspace=0.0, wspace=0.1)
        for i,ax in enumerate(axes.flatten()):
            ax.axis('off')
            if (i+1) % 9 == 0:
                ax.axis('on')
        self.logger.setLevel(100)
        old_level = self.logger.level
        for i in range(n_pic):
            axes[2*i,0].set_title('Real RGB')
            axes[2*i,1].set_title('Real Depth')
            axes[2*i,2].set_title('R-S Depth')
            axes[2*i,3].set_title('Img2Depth')
            axes[2*i,4].set_title('Cycle Depth A')
            axes[2*i,5].set_title('Real Norm')
            axes[2*i,6].set_title('R-S Norm')
            axes[2*i,7].set_title('Cycle Norm A')
            axes[2*i,8].set_title('Graph')
            
            axes[2*i+1,0].set_title('Syn RGB')
            axes[2*i+1,1].set_title('Syn Depth')
            axes[2*i+1,2].set_title('S-R Depth')
            axes[2*i+1,3].set_title('Cycle Depth B')
            axes[2*i+1,4].set_title('Cycle Depth B')
            axes[2*i+1,5].set_title('Syn Norm')
            axes[2*i+1,6].set_title('S-R Norm')
            axes[2*i+1,7].set_title('Cycle Norm B')
            axes[2*i+1,8].set_title('Graph')

            axes[2*i,0].imshow(A_imgs[i])
            axes[2*i,1].imshow(A_depth[i],cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=max_dist)
            axes[2*i,2].imshow(B_depth_fake[i],cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=max_dist)
            axes[2*i,3].imshow(A_img_depht[i],cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=max_dist)
            axes[2*i,4].imshow(A_img_depht[i],cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=max_dist)
            axes[2*i,5].imshow(A_norm[i])
            axes[2*i,6].imshow(B_norm_fake[i])
            axes[2*i,7].imshow(B_norm_fake[i])
            axes[2*i,8].plot(A_depth[i][100], label = 'Real Depth')
            axes[2*i,8].plot(B_depth_fake[i][100], label = 'R-S Depth')
            axes[2*i,8].legend()
            
            axes[2*i+1,0].imshow(B_imgs[i])
            axes[2*i+1,1].imshow(B_depth[i],cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=max_dist)
            axes[2*i+1,2].imshow(A_depth_fake[i],cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=max_dist)
            axes[2*i+1,3].imshow(B_depth_rec[i],cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=max_dist)
            axes[2*i+1,4].imshow(B_depth_rec[i],cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=max_dist)
            axes[2*i+1,5].imshow(B_norm[i])
            axes[2*i+1,6].imshow(A_norm_fake[i])
            axes[2*i+1,7].imshow(B_norm_rec[i])
            axes[2*i+1,8].plot(B_depth[i][100], label = 'Syn Depth')
            axes[2*i+1,8].plot(A_depth_fake[i][100], label = 'S-R Depth')
            axes[2*i+1,8].legend()
        self.logger.setLevel(old_level)                        
        return fig