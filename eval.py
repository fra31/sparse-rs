import os
import argparse
import torch
import torch.nn as nn

import numpy as np

import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision import models as torch_models

import sys
import time
from datetime import datetime

model_class_dict = {'pt_vgg': torch_models.vgg16_bn,
                    'pt_resnet': torch_models.resnet50,
                    }

class PretrainedModel():
    def __init__(self, modelname):
        model_pt = model_class_dict[modelname](pretrained=True)
        #model.eval()
        self.model = nn.DataParallel(model_pt.cuda())
        self.model.eval()
        self.mu = torch.Tensor([0.485, 0.456, 0.406]).float().view(1, 3, 1, 1).cuda()
        self.sigma = torch.Tensor([0.229, 0.224, 0.225]).float().view(1, 3, 1, 1).cuda()

    def predict(self, x):
        out = (x - self.mu) / self.sigma
        return self.model(out)

    def forward(self, x):
        out = (x - self.mu) / self.sigma
        return self.model(out)

    def __call__(self, x):
        return self.predict(x)

def random_target_classes(y_pred, n_classes):
    y = torch.zeros_like(y_pred)
    for counter in range(y_pred.shape[0]):
        l = list(range(n_classes))
        l.remove(y_pred[counter])
        t = torch.randint(0, len(l), size=[1])
        y[counter] = l[t] + 0

    return y.long()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='ImageNet')
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--norm', type=str, default='L0')
    parser.add_argument('--k', default=150., type=float)
    parser.add_argument('--n_restarts', type=int, default=1)
    parser.add_argument('--loss', type=str, default='margin')
    parser.add_argument('--model', default='pt_vgg', type=str)
    parser.add_argument('--n_ex', type=int, default=1000)
    parser.add_argument('--attack', type=str, default='sparse-rs')
    parser.add_argument('--n_queries', type=int, default=1000)
    parser.add_argument('--targeted', action='store_true')
    parser.add_argument('--target_class', type=int)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--constant_schedule', action='store_true')
    parser.add_argument('--alpha_init', type=float, default=.3)
    
    args = parser.parse_args()

    if args.data_path is None:
        args.data_path = "/scratch/maksym/imagenet/val_orig"
    
    args.eps = args.k + 0
    args.bs = args.n_ex + 0
    
    if args.dataset == 'ImageNet':
        # load pretrained model
        model = PretrainedModel(args.model)
        assert not model.model.training
        print(model.model.training)
        
        # load data
        IMAGENET_SL = 224
        IMAGENET_PATH = args.data_path
        imagenet = datasets.ImageFolder(IMAGENET_PATH,
                               transforms.Compose([
                                   transforms.Resize(IMAGENET_SL),
                                   transforms.CenterCrop(IMAGENET_SL),
                                   transforms.ToTensor()
                               ]))
        torch.manual_seed(0)
    
        test_loader = data.DataLoader(imagenet, batch_size=args.bs, shuffle=True, num_workers=0)
        
        testiter = iter(test_loader)
        x_test, y_test = next(testiter)
        
    if args.attack in ['sparse-rs']:
        # run Sparse-RS attacks
        if not os.path.exists('./results/{}/'.format(args.dataset)):
            os.makedirs('./results/{}/'.format(args.dataset))
        if not os.path.exists('./results/logs/'):
            os.makedirs('./results/logs/')
        
        if args.targeted or 'universal' in args.norm:
            args.loss = 'ce'
        param_run = '{}_{}_{}_1_{}_nqueries_{:.0f}_alphainit_{:.2f}_loss_{}_eps_{:.0f}_targeted_{}_targetclass_{}_seed_{:.0f}'.format(
            args.attack, args.norm, args.model, args.n_ex, args.n_queries, args.alpha_init,
            args.loss, args.eps, args.targeted, args.target_class, args.seed)
        if args.constant_schedule:
            param_run += '_constantpinit'
        
        from rs_attacks import RSAttack
        adversary = RSAttack(model, norm=args.norm, eps=int(args.eps), verbose=True, n_queries=args.n_queries,
            alpha_init=args.alpha_init, log_path='./results/{}/log_run_{}_{}.txt'.format('logs', str(datetime.now())[:-7], param_run),
            loss=args.loss, targeted=args.targeted, seed=args.seed, constant_schedule=args.constant_schedule)
        
        # set target classes
        if args.targeted and 'universal' in args.norm:
            if args.target_class is None:
                y_test = torch.ones_like(y_test) * torch.randint(1000, size=[1]).to(y_test.device)
            else:
                y_test = torch.ones_like(y_test) * args.target_class
            print('target labels', y_test)
        
        elif args.targeted:
            y_test = random_target_classes(y_test, 1000)
            print('target labels', y_test)
        
        bs = min(args.bs, 250)
        assert args.n_ex % args.bs == 0
        adv_complete = x_test.clone()
        qr_complete = torch.zeros([x_test.shape[0]]).cpu()
        pred = torch.zeros([0]).float().cpu()
        with torch.no_grad():
            # find points originally correctly classified
            for counter in range(x_test.shape[0] // bs):
                x_curr = x_test[counter * bs:(counter + 1) * bs].cuda()
                y_curr = y_test[counter * bs:(counter + 1) * bs].cuda()
                output = model(x_curr)
                if not args.targeted:
                    pred = torch.cat((pred, (output.max(1)[1] == y_curr).float().cpu()), dim=0)
                else:
                    pred = torch.cat((pred, (output.max(1)[1] != y_curr).float().cpu()), dim=0)
            
            adversary.logger.log('clean accuracy {:.2%}'.format(pred.mean()))
            
            n_batches = pred.sum() // bs + 1 if pred.sum() % bs != 0 else pred.sum() // bs
            n_batches = n_batches.long().item()
            ind_to_fool = (pred == 1).nonzero().squeeze()
            
            # run the attack
            pred_adv = pred.clone()
            for counter in range(n_batches):
                x_curr = x_test[ind_to_fool[counter * bs:(counter + 1) * bs]].cuda()
                y_curr = y_test[ind_to_fool[counter * bs:(counter + 1) * bs]].cuda()
                qr_curr, adv = adversary.perturb(x_curr, y_curr)
                
                output = model(adv.cuda())
                if not args.targeted:
                    acc_curr = (output.max(1)[1] == y_curr).float().cpu()
                else:
                    acc_curr = (output.max(1)[1] != y_curr).float().cpu()
                pred_adv[ind_to_fool[counter * bs:(counter + 1) * bs]] = acc_curr.clone()
                adv_complete[ind_to_fool[counter * bs:(counter + 1) * bs]] = adv.cpu().clone()
                qr_complete[ind_to_fool[counter * bs:(counter + 1) * bs]] = qr_curr.cpu().clone()
                
                print('batch {}/{} - {:.0f} of {} successfully perturbed'.format(
                    counter + 1, n_batches, x_curr.shape[0] - acc_curr.sum(), x_curr.shape[0]))
                
            adversary.logger.log('robust accuracy {:.2%}'.format(pred_adv.float().mean()))
            
            # check robust accuracy and other statistics
            acc = 0.
            for counter in range(x_test.shape[0] // bs):
                x_curr = adv_complete[counter * bs:(counter + 1) * bs].cuda()
                y_curr = y_test[counter * bs:(counter + 1) * bs].cuda()
                output = model(x_curr)
                if not args.targeted:
                    acc += (output.max(1)[1] == y_curr).float().sum().item()
                else:
                    acc += (output.max(1)[1] != y_curr).float().sum().item()
            
            adversary.logger.log('robust accuracy {:.2%}'.format(acc / args.n_ex))
            
            res = (adv_complete - x_test != 0.).max(dim=1)[0].sum(dim=(1, 2))
            adversary.logger.log('max L0 perturbation {:.0f} - nan in img {} - max img {:.5f} - min img {:.5f}'.format(
                res.max(), (adv_complete != adv_complete).sum(), adv_complete.max(), adv_complete.min()))
                
            ind_succ = (((pred_adv == 0.) * (pred == 1.)) == 1.).nonzero().squeeze()
            adversary.logger.log('success rate={:.0f}/{:.0f} ({:.2%}) - avg # queries {:.1f} - med # queries {:.1f}'.format(
                pred.sum() - pred_adv.sum(), pred.sum(), (pred.sum() - pred_adv.sum()).float() / pred.sum(),
                qr_complete[ind_succ].float().mean(), torch.median(qr_complete[ind_succ].float())))
            
            
            # save results depending on the threat model
            if args.norm in ['L0', 'patches', 'frames']:
                save_path = './results/{}/{}_{}_{}_1_{}_nqueries_{:.0f}_alphainit_{:.2f}_loss_{}_eps_{:.0f}_targeted_{}_seed_{:.0f}'.format(
                    args.dataset, args.attack, args.norm, args.model, args.n_ex, args.n_queries, args.alpha_init,
                    args.loss, args.eps, args.targeted, args.seed)
                if args.constant_schedule:
                    save_path += '_constantschedule'
                    
                torch.save({'adv': adv_complete, 'qr': qr_complete}, '{}.pth'.format(save_path))
                    
            elif args.norm in ['patches_universal']:
                # extract and save patch
                ind = (res > 0).nonzero().squeeze()[0]
                ind_patch = (((adv_complete[ind] - x_test[ind]).abs() > 0).max(0)[0] > 0).nonzero().squeeze()
                t = [ind_patch[:, 0].min().item(), ind_patch[:, 0].max().item(), ind_patch[:, 1].min().item(), ind_patch[:, 1].max().item()]
                loc = torch.tensor([t[0], t[2]])
                s = t[1] - t[0] + 1
                patch = adv_complete[ind, :, loc[0]:loc[0] + s, loc[1]:loc[1] + s].unsqueeze(0)
                
                torch.save({'adv': adv_complete, 'patch': patch},
                    './results/{}/{}_{}_{}_1_{}_nqueries_{:.0f}_alphainit_{:.2f}_loss_{}_eps_{:.0f}_targeted_{}.pth'.format(
                    args.dataset, args.attack, args.norm, args.model, args.n_ex, args.n_queries,
                    args.alpha_init, args.loss, args.eps, args.targeted))

            elif args.norm in ['frames_universal']:
                # extract and save frame and indeces of the perturbed pixels
                # to easily apply the frame to new images
                ind_img = (res > 0).nonzero().squeeze()[0]
                mask = torch.zeros(x_test.shape[-2:])
                s = int(args.eps)
                mask[:s] = 1.
                mask[-s:] = 1.
                mask[:, :s] = 1.
                mask[:, -s:] = 1.
                ind = (mask == 1.).nonzero().squeeze()
                frame = adv_complete[ind_img, :, ind[:, 0], ind[:, 1]]
                
                torch.save({'adv': adv_complete, 'frame': frame, 'ind': ind},
                    './results/{}/{}_{}_{}_1_{}_nqueries_{:.0f}_alphainit_{:.3f}_loss_{}_eps_{:.0f}_targeted_{}_targetclass_{}.pth'.format(
                    args.dataset, args.attack, args.norm, args.model, args.n_ex, args.n_queries,
                    alpha_init, args.loss, args.eps, args.targeted, args.target_class))
    
    else:
        raise ValueError('unknown attack')
            
