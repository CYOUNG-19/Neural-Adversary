import torch
import os
import sys
sys.path.append('./')
from model.resnet import return_resnet
from model.wrn import return_wrn
from model.vggnet import return_vgg
import torch.optim as optim
from dataset import return_dataloader
import torch.nn.functional as F
from attack import pgd
from torch.autograd import Variable
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
class SOLVER:
    def __init__(self, args):
        
        self.args = args
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.device = args.devices
        self.lr = args.lr
        self.save = args.save
        self.beta = args.beta
        self.tensorboard = args.tensorboard
        self.num_classes = args.num_classes
        self.dataset = args.dataset
        self.model = args.model
        self.num_iter = args.num_iter
        self.norm = args.norm
        self.epsilon = args.epsilon
        self.beta = args.beta
        self.alpha = args.alpha  
        self.method = args.method

        if self.dataset == 'cifar10' :
            self.input_size = 32
        elif args.dataset == 'cifar100':
            self.input_size = 32
        elif args.dataset == 'imagenet':
            self.input_size = 64
        else:
            raise ValueError(f"Unsupported model name: {args.dataset}")
        
        # network instance
        if args.model == 'resnet':
            self.network = return_resnet(self.num_classes, self.input_size).to(self.device)
        elif args.model == 'wrn':
            self.network = return_wrn(self.num_classes,self.dataset).to(self.device)
        elif args.model == 'vgg':
            self.network = return_vgg(self.num_classes, self.dataset).to(self.device)
        else:
            raise ValueError(f"Unsupported model name: {args.model}")


        # optimizer
        self.optim = optim.SGD(self.network.parameters(), lr = self.lr, momentum=0.9, weight_decay=5e-4)
        #self.optim = optim.Adam(self.network.parameters(), lr=0.005)
        
        # data
        if args.dataset == 'imagenet':
            self.train_loader = return_dataloader(args)
        else:
            self.train_loader, self.test_loader = return_dataloader(args)
        
        # tensorboard
        '''if self.tensorboard:
            self.writer = SummaryWriter()'''
    
    def adjust_learning_rate(self, epoch):
        """decrease the learning rate"""
        if self.model == 'vgg':
            lr =0.02
            if epoch >= 50:
                lr =  0.005   
            if epoch >= 100:
                lr =  0.001
            if epoch >= 150:
                lr =  0.0001

        else:
            lr = self.lr
            if epoch >= 50:
                lr =  0.01
            if epoch >= 100:
                lr =  0.001
            if epoch >= 150:
                lr =  0.0001


        for param_group in self.optim.param_groups:
            param_group['lr'] = lr
    
    def train_eval(self, epoch):
        """
        to salve time, we only test the attack during evaluation
        """
        
        self.network.eval()
        train_loss = 0.
        correct = 0.
        
        with torch.no_grad():
            for X, label in self.train_loader:
                X, label = X.to(self.device), label.to(self.device)
                output = self.network(X)
                train_loss += F.cross_entropy(output, label).item()
                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(label.view_as(pred)).sum().item()
        train_loss /= len(self.train_loader.dataset)
        print("TRAIN LOSS: ", round(train_loss, 5), "ACCURACY: ", round(correct/len(self.train_loader.dataset), 5))
        
    
'''class AT(SOLVER):
    def __init__(self, args):
        super().__init__(args)
        
    def loss(self, image, delta, label):
        
        # robust loss
        loss_robust = F.cross_entropy(self.network(image + delta), label)
        
        loss_total = loss_robust 
        
        return loss_total
    
    def train_epoch(self):
        self.network.train()
        all_loss = 0
        
        for batch_idx, (image, label) in enumerate(self.train_loader):
            image = image.to(self.device)
            label = label.to(self.device)
            
            self.optim.zero_grad()
            
            # generate attack
            delta = pgd(self.args, self.network, image, label)
            loss_total = self.loss(image, delta, label)
            all_loss += loss_total.item()
            
            loss_total.backward()
            self.optim.step()
        all_loss /= len(self.train_loader.dataset)
        return all_loss
    

class CLEAN(SOLVER):
    def __init__(self, args):
        super().__init__(args)
        
    def loss(self, image, label):
        
        # robust loss
        loss_total = F.cross_entropy(self.network(image), label)
        
        return loss_total
    
    def train_epoch(self):
        self.network.train()
        all_loss = 0
        
        for batch_idx, (image, label) in enumerate(self.train_loader):
            image = image.to(self.device)
            label = label.to(self.device)
            
            self.optim.zero_grad()
            loss_total = self.loss(image, label)
            all_loss += loss_total.item()
            
            loss_total.backward()
            self.optim.step()
        all_loss /= len(self.train_loader.dataset)
        return all_loss'''
    

class AT(SOLVER):
    def __init__(self, args):
        super().__init__(args)
        
    def pgd_loss(self,model,
                    device,
                    x_natural,
                    y,
                    step_size=0.003,
                    epsilon=0.031,
                    perturb_steps=10,
                    distance='l_2'):
        # define KL-loss
        criterion_kl = nn.KLDivLoss(reduction='sum')
        model.eval()
        batch_size = len(x_natural)
        # generate adversarial example
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(device).detach()
        if distance == 'l_inf':
            for _ in range(perturb_steps):
                x_adv.requires_grad_()
                with torch.enable_grad():
                    loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                            F.softmax(model(x_natural), dim=1))
                grad = torch.autograd.grad(loss_kl, [x_adv])[0]
                x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
                x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
                x_adv = torch.clamp(x_adv, 0.0, 1.0)
        elif distance == 'l_2':
            delta = 0.001 * torch.randn(x_natural.shape).to(device).detach()
            delta = Variable(delta.data, requires_grad=True)

            # Setup optimizers
            optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

            for _ in range(perturb_steps):
                adv = x_natural + delta

                # optimize
                optimizer_delta.zero_grad()
                with torch.enable_grad():
                    loss = (-1) * criterion_kl(F.log_softmax(model(adv), dim=1),
                                                F.softmax(model(x_natural), dim=1))
                loss.backward()
                # renorming gradient
                grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
                delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
                # avoid nan or inf if gradient is 0
                if (grad_norms == 0).any():
                    delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
                optimizer_delta.step()

                # projection
                delta.data.add_(x_natural)
                delta.data.clamp_(0, 1).sub_(x_natural)
                delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
            x_adv = Variable(x_natural + delta, requires_grad=False)
        else:
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
        model.train()

        x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
        # zero gradient
        self.optim.zero_grad()
        # calculate robust loss

        loss_robust = F.cross_entropy(self.network(x_adv), y)
        return loss_robust
    
    def pgd(self,model,
                    device,
                    x_natural,
                    step_size,
                    epsilon,
                    perturb_steps,
                    distance):
        # define KL-loss
        criterion_kl = nn.KLDivLoss(reduction='sum')
        model.eval()
        batch_size = len(x_natural)
        # generate adversarial example
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(device).detach()
        if distance == 'l_inf':
            for _ in range(perturb_steps):
                x_adv.requires_grad_()
                with torch.enable_grad():
                    loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                            F.softmax(model(x_natural), dim=1))
                grad = torch.autograd.grad(loss_kl, [x_adv])[0]
                x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
                x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
                x_adv = torch.clamp(x_adv, 0.0, 1.0)
        elif distance == 'l_2':
            delta = 0.001 * torch.randn(x_natural.shape).to(device).detach()
            delta = Variable(delta.data, requires_grad=True)

            # Setup optimizers
            optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

            for _ in range(perturb_steps):
                adv = x_natural + delta

                # optimize
                optimizer_delta.zero_grad()
                with torch.enable_grad():
                    loss = (-1) * criterion_kl(F.log_softmax(model(adv), dim=1),
                                                F.softmax(model(x_natural), dim=1))
                loss.backward()
                # renorming gradient
                grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
                delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
                # avoid nan or inf if gradient is 0
                if (grad_norms == 0).any():
                    delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
                optimizer_delta.step()

                # projection
                delta.data.add_(x_natural)
                delta.data.clamp_(0, 1).sub_(x_natural)
                delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
            x_adv = Variable(x_natural + delta, requires_grad=False)
        else:
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
        model.train()

        x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
        # zero gradient
        self.optim.zero_grad()

        return x_adv.detach()
    
    def trades_loss(self,model,
                    device,
                    x_natural,
                    y,
                    step_size=0.003,
                    epsilon=0.031,
                    perturb_steps=10,
                    beta=6.0,
                    distance='l_2'):
        # define KL-loss
        criterion_kl = nn.KLDivLoss(reduction='sum')
        model.eval()
        batch_size = len(x_natural)
        # generate adversarial example
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(device).detach()
        if distance == 'l_inf':
            for _ in range(perturb_steps):
                x_adv.requires_grad_()
                with torch.enable_grad():
                    loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                            F.softmax(model(x_natural), dim=1))
                grad = torch.autograd.grad(loss_kl, [x_adv])[0]
                x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
                x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
                x_adv = torch.clamp(x_adv, 0.0, 1.0)
        elif distance == 'l_2':
            delta = 0.001 * torch.randn(x_natural.shape).to(device).detach()
            delta = Variable(delta.data, requires_grad=True)

            # Setup optimizers
            optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

            for _ in range(perturb_steps):
                adv = x_natural + delta

                # optimize
                optimizer_delta.zero_grad()
                with torch.enable_grad():
                    loss = (-1) * criterion_kl(F.log_softmax(model(adv), dim=1),
                                                F.softmax(model(x_natural), dim=1))
                loss.backward()
                # renorming gradient
                grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
                delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
                # avoid nan or inf if gradient is 0
                if (grad_norms == 0).any():
                    delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
                optimizer_delta.step()

                # projection
                delta.data.add_(x_natural)
                delta.data.clamp_(0, 1).sub_(x_natural)
                delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
            x_adv = Variable(x_natural + delta, requires_grad=False)
        else:
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
        model.train()

        x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
        # zero gradient
        self.optim.zero_grad()
        # calculate robust loss
        logits = model(x_natural)
        loss_natural = F.cross_entropy(logits, y)
        loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                                        F.softmax(model(x_natural), dim=1))
        loss = loss_natural + beta * loss_robust
        return loss

    def MART_loss(self, model, device, x_natural, target, step_size, epsilon, perturb_steps, distance, beta):
        # Based on the repo MART https://github.com/YisenWang/MART
        adv_logits = model(self.pgd(model,
                    device,
                    x_natural,
                    step_size,
                    epsilon,
                    perturb_steps,
                    distance))
        natural_logits = model(x_natural)

        kl = nn.KLDivLoss(reduction='none')
        batch_size = len(target)
        adv_probs = F.softmax(adv_logits, dim=1)
        tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]
        new_y = torch.where(tmp1[:, -1] == target, tmp1[:, -2], tmp1[:, -1])
        loss_adv = F.cross_entropy(adv_logits, target) + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y)
        nat_probs = F.softmax(natural_logits, dim=1)
        true_probs = torch.gather(nat_probs, 1, (target.unsqueeze(1)).long()).squeeze()
        loss_robust = (1.0 / batch_size) * torch.sum(
            torch.sum(kl(torch.log(adv_probs + 1e-12), nat_probs), dim=1) * (1.0000001 - true_probs))
        loss = loss_adv + float(beta) * loss_robust
        return loss


    def clean_loss(self, image, label):
        
        # robust loss
        loss_total = F.cross_entropy(self.network(image), label)
        
        return loss_total

    def train_epoch(self):
        self.network.train()
        all_loss = 0
        
        for batch_idx, (image, label) in enumerate(self.train_loader):
            image = image.to(self.device)
            label = label.to(self.device)
            
            self.optim.zero_grad()
            
            # generate attack
            if self.method == 'at':
                loss_total = self.pgd_loss(self.network,
                        self.device,
                        image,
                        label,
                        self.alpha,
                        self.epsilon,
                        self.num_iter,
                        self.norm)
                
            elif self.method == 'trades':
                loss_total = self.trades_loss(self.network,
                        self.device,
                        image,
                        label,
                        self.alpha,
                        self.epsilon,
                        self.num_iter,
                        self.beta,
                        self.norm)
                
            elif self.method == 'clean':
                loss_total = self.clean_loss(image, label)
                
            elif self.method == 'mart':
                loss_total = self.MART_loss(self.network,
                        self.device,
                        image,
                        label, 
                        self.alpha,
                        self.epsilon, 
                        self.num_iter,
                        self.norm, 
                        self.beta)
            all_loss += loss_total.item()
            loss_total.backward()
            self.optim.step()
        all_loss /= len(self.train_loader.dataset)
        return all_loss


