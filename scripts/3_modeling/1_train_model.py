from argparse import ArgumentParser, Namespace

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Pytorch lightning
import pytorch_lightning as pl
from pl_examples import cli_lightning_logo
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# My Modules
from utils.model_dataset import GlacierDataset
import utils.transforms as custom_transforms
import utils.loss_functions as losses
from utils.unet_models import U_Net, R2U_Net, AttU_Net, R2AttU_Net, AttU_Net_big
import segmentation_models_pytorch as smp
from crfseg import CRF


class GlacierModel(pl.LightningModule):
    """
    
    """

    def __init__(
        self,
        train_dir: str,
        val_dir: str,
        max_epochs: int,
        lr: float,
        batch_size: int,
        gpus: int,
        precision: int,
        early_stop_callback: list = [],
        auto_scale_batch_size: str = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
                
        # Parameters
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.batch_size = batch_size
        self.learning_rate = lr

        # Define the network
        self.net = smp.MAnet(
            encoder_name = "resnet50",
            encoder_weights = 'imagenet',
            in_channels = 9,
            classes = 3
            )
                
        # Load the two data loaders    
        self.trainset,  self.validset = self.get_glacier_datasets()
       
    def get_glacier_datasets (self):
        
        # Define the transforms that need to be use
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
        training_dataset = GlacierDataset(self.train_dir, transform=data_transforms)
        val_dataset = GlacierDataset(self.val_dir, transform=None)
    
        return training_dataset, val_dataset

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_nb):
        img, mask = batch
        img = img.float()
        mask = mask.long()
        out = self(img)
        loss = losses.jaccard_loss(out, mask)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        img, mask = batch
        img = img.float()
        mask = mask.long()
        out = self(img)
        loss_val = losses.jaccard_loss(out, mask)
        loss_score = losses.jaccard_score(out, mask)
        return {'val_loss': loss_val, 'val_score': loss_score}

    def validation_epoch_end(self, outputs):
        loss_val = torch.stack([x['val_loss'] for x in outputs]).mean()
        loss_score = torch.stack([x['val_score'] for x in outputs]).mean()
        self.log('val_loss', loss_val, on_epoch=True, prog_bar=True)
        return None

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=25000)
        return [opt], [sched]

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True, num_workers=16)

    def val_dataloader(self):
        return DataLoader(self.validset, batch_size=self.batch_size, shuffle=False, num_workers=16)

    @staticmethod
    def add_model_specific_args(parent_parser):
        
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument("--train_dir", type=str, help="path where the training dataset is stored")
        parser.add_argument("--val_dir", type=str, help="path where the validation dataset is stored")
        parser.add_argument("--max_epochs", type=int, default=50, help="maximum number of epochs to train the model")
        parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
        parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
        parser.add_argument('--gpus', type=int, default=1, help='number of CPUs to utilize')
        parser.add_argument('--precision', type=str, default=32, help='Mixed Precision Setting')
        parser.add_argument('--auto_scale_batch_size', type=str, default='binsearch', help='search strat for finding optimal batchsize')
        
        return parser

def get_optimal_batch_size(params):
    ''' Identify the optimal batch size'''
    
    # Define the trainer    
    trainer = pl.Trainer.from_argparse_args(hparams)
    
    # Create the model
    model = GlacierModel(**vars(params))
    
    # Run the tuning
    print('\nTuning batch size:')
    trainer.tune(model)
    
    # Get the output batch size
    out_batch_size = int(model.batch_size * 0.90)
    
    # Stop an obnoxious memory leak
    del model
    torch.cuda.empty_cache()
    
    return out_batch_size

def get_optimal_learning_rate(params):
    
    # Define the trainer    
    trainer = pl.Trainer.from_argparse_args(hparams)
    
    # Create the model
    model = GlacierModel(**vars(params))
    
    '''Using the learning rate range test to identify an optimal learning rate'''
    # Run learning rate finder
    print('\nTuning learning rate:')
    lr_finder = trainer.tuner.lr_find(model)
    
    # Pick point based on plot, or get suggestion
    new_lr = lr_finder.suggestion()
    
    return new_lr

def main(hparams: Namespace):
    
    # ------------------------
    # 1 Add early stopping
    # ------------------------

    # Define the early stopping callback
    #early_stop = EarlyStopping(
    #    monitor='val_loss',
    #    min_delta=0.0005,
    #    patience=2,
    #    verbose=False,
    #    mode='min'
    #)
    #hparams.early_stop_callback = [early_stop]
    
    # # ------------------------
    # # 3 GET THE OPTIMAL BATCH SIZE
    # # ------------------------
    
    # hparams.batch_size = get_optimal_batch_size(hparams)
    # print('Optimal batch size:', hparams.batch_size)
    
    # ------------------------
    # 4 SELECT THE LEARNING RATE
    # ------------------------
    # update hparams of the model
    #hparams.lr = get_optimal_learning_rate (hparams)
    #print('Optimal Learning Rate:', hparams.lr)
	
    # ------------------------
    # 3 START TRAINING
    # ------------------------
    # try:
    model = GlacierModel(**vars(hparams))
    trainer = pl.Trainer.from_argparse_args(hparams)
    trainer.fit(model)
    # except RuntimeError as e:
    #     print('OOM Error because WSL2 blows', e)
    
    # finally:
        
    #     # Lower the batch size
    #     hparams.batch_size = int(hparams.batch_size * 0.8)
    #     print('New suboptimal batch size:', hparams.batch_size)
        
    #     # Attempt to train again
    #     model = GlacierModel(**vars(hparams))
    #     trainer = pl.Trainer.from_argparse_args(hparams)
    #     trainer.fit(model)
        
    return None

if __name__ == '__main__':
    
    parser = ArgumentParser(add_help=False)
    parser = GlacierModel.add_model_specific_args(parser)
    hparams = parser.parse_args()

    main(hparams)