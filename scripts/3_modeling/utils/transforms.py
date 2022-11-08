import torch
import random

class AddGaussianNoise(object):
    
    def __init__(self, mean=0.0, std=0.01):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class NullTransform(object):
    """Placeholder transformation. Does not change the input at all"""
    def __init__(self):
        return

    def __call__(self, in_tensor):
        """
        Args:
            img (Torch Tensor): Image to be flipped.

        Returns:
            Torch Tensor: The saem input.
        """
        return in_tensor
     
    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class RotateLeft2DTensor(object):
    """Horizontally flip the given 4D Torch Tensor randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self):
        return None
    
    def __call__(self, in_tensor):
        """
        Args:
            img (Torch Tensor): Image to be flipped.

        Returns:
            Torch Tensor: Randomly flipped image.
        """
        return torch.rot90(in_tensor, 1, [2,1])

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RotateRight2DTensor(object):
    """Horizontally flip the given 4D Torch Tensor randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self):
        return None

    def __call__(self, in_tensor):
        """
        Args:
            img (Torch Tensor): Image to be flipped.

        Returns:
            Torch Tensor: Randomly flipped image.
        """

        return torch.rot90(in_tensor, 1, [2,1])
 
    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class VerticalFlip2DTensor(object):
    """Vertically flip the given 4D Torch Tensor randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self):
        return None

    def __call__(self, in_tensor):
        """
        Args:
            img (Torch Tensor): 4D array to be flipped along the 3rd axis.

        Returns:
            Torch Tensor: Randomly flipped image.
        """
        return torch.rot90(in_tensor, 2, [2,1])

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)
    
    
if __name__ == '__main__':

    # Create a random torch tensor
    test_tensor = torch.rand(1,3,3)
    print(test_tensor, '\n')
    
    # Generate the transformers
    test_left = RotateLeft2DTensor()
    test_right = RotateRight2DTensor()
    test_vertical = VerticalFlip2DTensor()
    
    # Test the flips
    left = test_left(test_tensor)
    print(left, '\n') 
    
    right = test_right(test_tensor)
    print(right, '\n') 
    
    vertical = test_vertical(test_tensor)
    print(vertical, '\n')    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    