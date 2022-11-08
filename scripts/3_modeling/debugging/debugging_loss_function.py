import torch
from torch.autograd import Variable
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from torch.nn import functional as F

# My Modules
from utils.model_dataset import GlacierDataset
from utils.model_dataset import GlacierDatasetBinary
import utils.loss_functions as losses
from utils.unet_models import U_Net, R2U_Net, AttU_Net, R2AttU_Net

#def tanimoto_complement_loss () :
#    """
#    An implementation of the tanimoto loss wityh Complement presented in the ResUNET-a paper
#    """
#    
#    # Need to apply an activation function to the logits
#    
#    return None

def jaccard_loss(logits, true,  eps=1e-7):
    """Computes the Jaccard loss, a.k.a the IoU loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the jaccard loss so we
    return the negated jaccard loss.
    Args:
        true: a tensor of shape [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        jacc_loss: the Jaccard loss.
        
    """
    
    # Check that the input is four dimensional [B, C, H, W]
    logit_dims = logits.shape
    if len(logit_dims) != 4:
        raise ValueError(
                "Input to jaccard loss function has {} dimensions, instead of 4.".format(len(logit_dims))
                )
    
    # Check that the number if classes is >1
    num_classes = logit_dims[1]
    if num_classes<= 1:
        raise ValueError("Input to jaccard loss function has {} channels. Inputs must have 2 (binary classification) or more channels.".format(num_classes))
        
    true_1_hot = torch.eye(num_classes)[true.squeeze(1).long()]
    true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
    probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    union = cardinality - intersection
    jacc_loss = (intersection / (union + eps)).mean()
    
    return (1 - jacc_loss)

if __name__ == '__main__':
    
    # Define the path to the training, testing, dev sets
    training_dir = "D:\\glacier_tensors\\training_data"
    
    # Define the batch size
    batch_size = 8
    
    print('Loading model...')
    net = AttU_Net(10, 2).to("cuda:0")
    net.eval()
    
    # Create the dataloader
    training_dataset = GlacierDatasetBinary(training_dir)
    train_loader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # Get a single example from the data loader
    inputs, labels = next(iter(train_loader))
    
    # Make sure the loss can be computed on the GPU
    images = Variable(inputs).cuda()
    labels = Variable(labels).cuda().long()

    # Get the prediction
    logits = net(images)
    
    print("Logits shape:", logits.shape)
    
    # Get the prediction over each of the values'
    print(jaccard_loss(logits, labels))
    

    # Compute the loss    
    
    
    
    
    
    
    
    