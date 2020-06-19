import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path_data', type=str)

args = parser.parse_args()

if args.path_data is None:
    path_data = './results/ImageNet/sparse-rs_L0_pt_vgg_1_50_nqueries_1000_alphainit_0.30_loss_margin_eps_150_targeted_False_seed_0.pth'
else:
    path_data = args.path_data

data = torch.load(path_data)
if 'qr' in list(data.keys()):
    imgs, qr = data['adv'].cpu(), data['qr'].cpu()
else:
    imgs, qr = data['adv'].cpu(), torch.arange(1, data['adv'].shape[0])

nqueries = 100000
ind = ((qr > 0) * (qr < nqueries)).nonzero().squeeze()
imgs_to_show = imgs[ind].permute(0, 2, 3, 1).cpu().numpy()
if imgs_to_show.shape[-1] == 1:
    imgs_to_show = np.tile(imgs_to_show, (1, 1, 1, 3))
qr_to_show = qr[ind]

qr_to_show, ind = qr_to_show.sort(descending=True)
imgs_to_show = imgs_to_show[ind]

w = 10
h = 10
fig = plt.figure(figsize=(20, 12))

columns = 10
rows = 5

# ax enables access to manipulate each of subplots
ax = []

init_pos = 0
if 'patch' in list(data.keys()):
    ax.append( fig.add_subplot(rows, columns, 1) )
    ax[-1].set_title('patch')
    ax[-1].get_xaxis().set_ticks([])
    ax[-1].get_yaxis().set_ticks([])
    ax[-1].axis('off')
    print(data['patch'].shape)
    s = int(float(path_data.split('eps_')[1].split('_')[0]) ** .5)
    patch = data['patch'].squeeze().view(-1, s, s)
    plt.imshow(patch.permute(1, 2, 0).cpu().numpy(), interpolation='none')
    init_pos = 1
    
for i in range(init_pos, columns*rows):
    if i < imgs_to_show.shape[0]:
        ax.append( fig.add_subplot(rows, columns, i+1) )
        ax[-1].set_title('qr = {:.0f}'.format(qr_to_show[i].item()))
        ax[-1].get_xaxis().set_ticks([])
        ax[-1].get_yaxis().set_ticks([])
        ax[-1].axis('off')
        plt.imshow(imgs_to_show[i], interpolation='none')
    
if not os.path.exists('./results/plots/'):
    os.makedirs('./results/plots/')

#plt.show()
plt.savefig('./results/plots/pl_{}.pdf'.format(path_data.split('/')[-1][:-4]))
