from argparse import ArgumentParser

import torch
import torchvision.transforms as transforms
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

# Import custom modules
import utils.transforms as custom_transforms
from utils.model_dataset import GlacierDataset
from utils.unet_models import U_Net, AttU_Net



def main():
    
    parser = ArgumentParser('DDP usage example')
    
    # Parameters about DDP usage
    parser.add_argument('--local_rank', type=int, default=-1, metavar='N', help='Local process rank.') 
    
    # Parameters about training data
    parser.add_argument('--train_dir', type=str, default="C:\\temp_ssd_space\\tensors\\training_data", metavar='N', help='Directory of the training set.')
    
    # Parameters about training loop
    parser.add_argument('--batch_size', type=int, default=8, metavar='N', help='The size of each minibatch.')
    parser.add_argument('--epochs', type=int, default=4, metavar='N', help='The number of epochs to train the model.')
    args = parser.parse_args()

    # keep track of whether the current process is the `master` process (totally optional, but I find it useful for data laoding, logging, etc.)
    args.is_master = args.local_rank == 0

    # set the device
    args.device = torch.cuda.device(args.local_rank)

    # initialize PyTorch distributed using environment variables (you could also do this more explicitly by 
    # specifying `rank` and `world_size`, but I find using environment variables makes it so that you can 
    # easily use the same script on different machines)
    dist.init_process_group(
        backend='gloo', 
        init_method='env://')
    torch.cuda.set_device(args.local_rank)

    # set the seed for all GPUs (also make sure to set the seed for random, numpy, etc.)
    torch.cuda.manual_seed_all(SEED)

    # initialize your model (BERT in this example)
    model = U_Net(10, 3)

    # send your model to GPU
    model = model.to(args.device)

    # initialize distributed data parallel (DDP)
    model = DDP(
        model,
        device_ids=[args.local_rank],
        output_device=args.local_rank
    )

    # initialize your dataset
    data_transforms = transforms.Compose([
            transforms.RandomChoice([
                custom_transforms.NullTransform(), # No transformation
                transforms.Compose([ # -90 degrees
                    custom_transforms.RotateLeft2DTensor(),
                ]),
                transforms.Compose([ # +90
                    custom_transforms.RotateRight2DTensor(),
                ]),
                transforms.Compose([ # +180
                    custom_transforms.RotateRight2DTensor(),
                    custom_transforms.RotateRight2DTensor()
                ]),
                transforms.Compose([ # Flip image
                    custom_transforms.VerticalFlip2DTensor()
                ]),
                transforms.Compose([ # Flip image + -90 degrees
                    custom_transforms.VerticalFlip2DTensor(),
                    custom_transforms.RotateLeft2DTensor()
                ]),
                transforms.Compose([ # Flip image + 90 degrees
                    custom_transforms.VerticalFlip2DTensor(),
                    custom_transforms.RotateRight2DTensor()
                ]),
            ])            
        ])
        
    # Load in the training datasets
    dataset = GlacierDataset(args.train_dir, transform=data_transforms)

    # initialize the DistributedSampler
    sampler = DistributedSampler(dataset)

    # initialize the dataloader
    dataloader = DataLoader(
        dataset=dataset,
        sampler=sampler,
        batch_size=args.batch_size,
    )

    # start your training!
    for epoch in range(args.epochs):
        
        # put model in train mode
        model.train()

        # let all processes sync up before starting with a new epoch of training
        dist.barrier()

        for step, batch in enumerate(dataloader):
            
            # get the inputs
            inputs, labels = data
            
            # Wrap them in Variable
            inputs = Variable(inputs).to(args.device)
            labels = Variable(labels).to(args.device).long()

            # Run the forward passorward pass
            outputs = model(inputs)

            # Compute the loss
            loss = criterion(outputs, labels)
            
            # Backpropagate
            loss.backwards()
            

# Global variables
SEED = 42

if __name__ == '__main__':
    
    main()
    