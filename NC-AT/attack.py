import torch
import copy
import torch.nn.functional as F

def initialize_feature(args, X):
    """
    initialize the attack vector in the feature domain
    """

    # randomly perturbation initialization
    delta = torch.rand_like(X, requires_grad=True) #0-1
    # Rescale into [-epsilon,epsilon]
    delta.data = delta.data * 2 * args.epsilon - args.epsilon
    # Make sure that the perturbed image is in [0,-1]
    delta.data.clamp_(min = 0-X, max = 1-X)

    # # zero perturbation initialization
    # delta = torch.zeros_like(X, requires_grad=True)
    
    return delta

def pgd(args, network, X, label):
    """
    perturbing the original image to maximize the classification error
    """
    
    network = copy.deepcopy(network)
    network.eval()  # disable BN and dropout
    
    epsilon = args.epsilon   # Perturbation strength   
    alpha = args.alpha       # lr
    num_iter = args.num_iter # Iteration number
    norm = args.norm
    # Initialization
    #delta = initialize_feature(args, X)  # random initialization, same to the original paper
    delta = 0.001 * torch.randn_like(X).detach()
    delta.requires_grad_()
    
    for t in range(num_iter):
        with torch.enable_grad():
            cross_entropy_loss = F.cross_entropy(network(X.detach() + delta), label)
        
        # maximize the classification error
        cross_entropy_loss.backward()
        if norm == 'l_2':
            grad_norm = torch.norm(delta.grad.detach(), p=2)
            grad_normalized = delta.grad.detach() / (grad_norm + 1e-10)  # Avoid division by zero
            
            # Update delta using the normalized gradients
            delta.data = (delta.detach() + alpha * grad_normalized)
            
            # Project delta onto the L2 ball
            norm_delta = torch.norm(delta.data, p=2)
            # Apply the projection only if the norm is greater than epsilon
            if norm_delta > epsilon:
                delta.data *= epsilon / norm_delta
            
            # Ensure delta is still in the valid image range
            delta.data.clamp_(min=(0 - X), max=(1 - X))
            
            # Zero out the gradient for the next iteration
            delta.grad.zero_()

        elif norm == 'l_inf':
            delta.data = (delta.detach() + alpha*delta.grad.detach().sign())
            
            # clamp within the [-epsilon,epsilon]
            delta.data.clamp_(min = -epsilon, max = epsilon)
            
            # clamp within [0,1]
            delta.data.clamp_(min = 0-X, max = 1-X)
            
            # zero out the gradient
            delta.grad.zero_()

    return delta.detach()