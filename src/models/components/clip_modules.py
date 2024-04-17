import torch.nn as nn
import torch

class TextCLIP(nn.Module):
    """
    TextCLIP is a class that wraps a model to handle text encoding.

    Args:
        model (nn.Module): A text encoding model.

    Attributes:
        model (nn.Module): The text encoding model.

    """
    def __init__(self, model):
        super(TextCLIP, self).__init__()
        self.model = model

    def forward(self, text):
        return self.model.encode_text(text)


class ImageCLIP(nn.Module):
    """
    A class representing an ImageCLIP model.

    Args:
        model: The model used for encoding images.

    Attributes:
        model: The model used for encoding images.

    Methods:
        forward(image): Encodes the given image using the model.

    """
    def __init__(self, model):
        super(ImageCLIP, self).__init__()
        self.model = model

    def forward(self, image):
        return self.model.encode_image(image)


class FrameAggregation(nn.Module):
    """
    FrameAggregation

    This class is used for aggregating frames in a video sequence using various methods.

    Attributes:
        method (str): The method to use for frame aggregation. The default value is "avg".

    Methods:
        forward(x):
            Aggregates the frames in the input tensor using the specified method.

    Example Usage:
        aggregation = FrameAggregation(method="avg")
        x = torch.randn(32, 10, 512)  # Batch of 32 videos with 10 frames each
        result = aggregation.forward(x)

        aggregation = FrameAggregation(method="max")
        x = torch.randn(16, 8, 256)  # Batch of 16 videos with 8 frames each
        result = aggregation.forward(x)
    """
    def __init__(self, method="avg"):
        super(FrameAggregation, self).__init__()
        self.method = method

    def forward(self, x):
        if self.method == "avg":
            return x.mean(dim=1)
        elif self.method == 'max':
            pooled_embedding, _ = torch.max(x, dim=1)
            return pooled_embedding
        elif self.method == 'min':
            pooled_embedding, _ = torch.min(x, dim=1)
            return pooled_embedding
        else:
            return x.mean(dim=1)
