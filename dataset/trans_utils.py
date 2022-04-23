import torch
import torch.nn.functional as F
# from torchvision.utils import _log_api_usage_once


class RandomPosterize(torch.nn.Module):
    """Posterize the image randomly with a given probability by reducing the
    number of bits for each color channel. If the image is torch Tensor, it should be of type torch.uint8,
    and it is expected to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".

    Args:
        bits (int): number of bits to keep for each channel (0-8)
        p (float): probability of the image being posterized. Default value is 0.5
    """

    def __init__(self, bits, p=0.5):
        super().__init__()
        # _log_api_usage_once(self)
        self.bits = bits
        self.p = p

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be posterized.

        Returns:
            PIL Image or Tensor: Randomly posterized image.
        """
        if torch.rand(1).item() < self.p:
            return F.posterize(img, self.bits)
        return img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(bits={self.bits},p={self.p})"