import torch
import os
import sys
sys.path.append('./')
import torch.nn.functional as F
from trainer import AT
import argparse
import pandas as pd
from scipy.sparse.linalg import svds
from attack import pgd
import numpy as np
import time
from torch.optim.lr_scheduler import StepLR

class graphs:
    def __init__(self):
        self.accuracy     = []
        self.loss         = []
        self.reg_loss     = []

        # NC1
        self.Sw_invSb     = []

        # NC2
        self.norm_M_CoV   = []
        self.norm_W_CoV   = []
        self.cos_M        = []
        self.cos_W        = []
        #ETF ditance
        self.m   = [] 


        # NC3
        self.W_M_dist     = []
            
        # NC4
        self.NCC_mismatch = []

        # Decomposition
        self.MSE_wd_features = []
        self.LNC1 = []
        self.LNC23 = []
        self.Lperp = []

class features:
        pass

def hook(self, input, output):
    features.value = input[0].clone()

def analysis_adv(graphs, model, device,  loader, C, classifier, features, args):
    model.eval()
    with torch.enable_grad():

        N             = [0 for _ in range(C)]
        mean          = [0 for _ in range(C)]
        Sw            = 0

        loss          = 0
        net_correct   = 0
        NCC_match_net = 0

        for computation in ['Mean','Cov']:
            #pbar = tqdm(total=len(loader), position=0, leave=True)
            for batch_idx, (data, target) in enumerate(loader, start=1):

                data, target = data.to(device), target.to(device)
                if args.method == 'clean':
                    output = model(data)
                else:
                    output_adv = trainer.pgd(model, args.devices, data, args.alpha, args.epsilon, args.num_iter, args.norm)
                    output = model(output_adv)

                h = features.value.data.view(data.shape[0],-1) # B CHW

                for c in range(C):
                    # features belonging to class c
                    idxs = (target == c).nonzero(as_tuple=True)[0]
                    
                    if len(idxs) == 0: # If no class-c in this batch
                        continue

                    h_c = h[idxs,:] # B CHW

                    if computation == 'Mean':
                        # update class means
                        mean[c] += torch.sum(h_c, dim=0) # CHW
                        N[c] += h_c.shape[0]
                        
                    elif computation == 'Cov':
                        # update within-class cov

                        z = h_c - mean[c].unsqueeze(0) # B CHW
                        cov = torch.matmul(z.unsqueeze(-1), # B CHW 1
                                            z.unsqueeze(1))  # B 1 CHW
                        Sw += torch.sum(cov, dim=0)

                        # during calculation of within-class covariance, calculate:
                        # 1) network's accuracy
                        net_pred = torch.argmax(output[idxs,:], dim=1)
                        net_correct += sum(net_pred==target[idxs]).item()

                        # 2) agreement between prediction and nearest class center
                        NCC_scores = torch.stack([torch.norm(h_c[i,:] - M.T,dim=1) \
                                                    for i in range(h_c.shape[0])])
                        NCC_pred = torch.argmin(NCC_scores, dim=1)
                        NCC_match_net += sum(NCC_pred==net_pred).item()

            
            if computation == 'Mean':
                for c in range(C):
                    mean[c] /= N[c]
                    M = torch.stack(mean).T
                
            elif computation == 'Cov':
                Sw /= sum(N)

        # global mean
        muG = torch.mean(M, dim=1, keepdim=True) # CHW 1
        
        # between-class covariance
        M_ = M - muG
        Sb = torch.matmul(M_, M_.T) / C
        # avg norm
        W  = classifier.weight
        M_norms = torch.norm(M_,  dim=0)
        W_norms = torch.norm(W.T, dim=0)
        graphs.norm_M_CoV.append((torch.std(M_norms)/torch.mean(M_norms)).item())
        graphs.norm_W_CoV.append((torch.std(W_norms)/torch.mean(W_norms)).item())
        graphs.NCC_mismatch.append(1-NCC_match_net/sum(N))

        graphs.m.append(M_.T)


        # tr{Sw Sb^-1}
        Sw = Sw.cpu().numpy()
        Sb = Sb.cpu().numpy()
        '''if np.any(np.isnan(Sb)) or np.any(np.isinf(Sb)):
            Sb = np.nan_to_num(Sb, nan=0.0, posinf=1.0, neginf=-1.0)'''
        print('C,Sb.shape',C,Sb.shape)
        eigvec, eigval, _ = svds(Sb, k=C-1)
        inv_Sb = eigvec @ np.diag(eigval**(-1)) @ eigvec.T 
        graphs.Sw_invSb.append(np.trace(Sw @ inv_Sb) / C)

        # ||W^T - M_||
        normalized_M = M_ / torch.norm(M_,'fro')
        normalized_W = W.T / torch.norm(W.T,'fro')
        graphs.W_M_dist.append((torch.norm(normalized_W - normalized_M)**2).item())

        # mutual coherence
        def coherence(V): 
            G = V.T @ V
            G += torch.ones((C,C),device=device) / (C-1)
            G -= torch.diag(torch.diag(G))
            return torch.norm(G,1).item() / (C*(C-1))

        graphs.cos_M.append(coherence(M_/M_norms))
        graphs.cos_W.append(coherence(W.T/W_norms))


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    #model
    parser.add_argument("--model", type = str, default = 'resnet')# resnet wrn vgg

    #dataset
    parser.add_argument("--dataset", type = str, default = 'cifar100')# cifar10 cifar100 imagenet (tiny)
    parser.add_argument("--num_classes", type = int, default = 100)#10 100 200

    #method
    parser.add_argument("--method", type = str, default = 'at')#clean at trades

    # train
    parser.add_argument("--batch_size", type = int, default = 128)
    parser.add_argument("--epochs", type = int, default = 200)
    parser.add_argument("--devices", type = str, default = "cuda:4")
    parser.add_argument("--lr", type = float, default = 0.05)
    parser.add_argument("--beta", type = float, default = 1)
    parser.add_argument("--save", type = bool, default = False)
    parser.add_argument("--seed", type = int, default = 21)
    parser.add_argument("--small_set", type = bool, default = False)
    parser.add_argument("--tensorboard", type = bool, default = False)
    
    # attack params
    parser.add_argument("--epsilon", type = float, default = 0.031)
    parser.add_argument("--alpha", type = float, default = 0.007)#0.007 0.055
    parser.add_argument("--num_iter", type = int, default = 10)
    parser.add_argument("--norm", type = str, default = 'l_inf') # l_2, l_inf
    
    args = parser.parse_args()
    if args.dataset == 'cifar10':
        args.num_classes = 10
    elif args.dataset == 'cifar100':
        args.num_classes = 100
    elif args.dataset == 'imagenet':
        args.num_classes = 200
    print(args)
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    trainer = AT(args)

    graphs = graphs()
    columns = ['train_loss', 'NC1', 'NC2_M_Norm', 'NC2_M_cos', 'NC2_W_Norm', 'NC2_W_cos', 'NC3', 'NC4', 'ETF_length']
    print_result = pd.DataFrame(columns=columns)

    # model saving address
    path = f'trained_model/{args.method}/{args.model}/{args.dataset}/{args.norm}'
    if not os.path.exists(path):
        os.makedirs(path)

    if args.model == 'wrn':
        classifier = trainer.network.linear
    elif args.model == 'resnet':
        classifier = trainer.network.fc
    else:
        classifier = trainer.network.classifier.fc3
    classifier.register_forward_hook(hook)

    analysis_adv(graphs, trainer.network, args.devices, trainer.train_loader, args.num_classes, classifier, features, args)
    print_result.loc[0] = [0, graphs.Sw_invSb[-1], graphs.norm_M_CoV[-1], graphs.cos_M[-1], graphs.norm_W_CoV[-1], graphs.cos_W[-1], graphs.W_M_dist[-1], graphs.NCC_mismatch[-1], 0]
    print_result.to_csv(os.path.join(path, "result_"+ str(args.epsilon) +".csv"))

    for i in range(1,args.epochs+1):
        #trainer.train_eval(i)

        trainer.adjust_learning_rate(i)
        print('epoch: ', i, 'learning rate: ', round(trainer.optim.param_groups[0]['lr'],5))
        loss = trainer.train_epoch()
        
        if (i % 5) == 1 or i == args.epochs:
            analysis_adv(graphs, trainer.network, args.devices, trainer.train_loader, args.num_classes, classifier, features, args)
            A = graphs.m[-1:]
            numpy_array = np.array([item.cpu().detach().numpy() for item in A])
            # 将 numpy 数组转换为 PyTorch 张量
            inputs = torch.tensor(numpy_array).to(args.devices)
            A = inputs.cpu().numpy()
            norm_along_axis1 = np.linalg.norm(A[0], ord=2, axis=1) 
            a=0
            for j in range(args.num_classes):
                a += norm_along_axis1[j]
            print_result.loc[i] = [loss, graphs.Sw_invSb[-1], graphs.norm_M_CoV[-1], graphs.cos_M[-1], graphs.norm_W_CoV[-1], graphs.cos_W[-1], graphs.W_M_dist[-1], graphs.NCC_mismatch[-1], a/args.num_classes]
            print_result.to_csv(os.path.join(path, "result_" + str(args.epsilon) +  ".csv"))#'_' +str(args.num_iter) +
        

    torch.save(trainer.network.state_dict(), f'{path}/{args.epsilon}.pth')
    