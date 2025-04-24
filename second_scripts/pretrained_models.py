import torchvision.models as models
from torch import nn
from torchvision import transforms

def get_pretrained_model(model_name: str="resnet18", num_classes: int=5) -> nn.Module:
    """
    Load a pretrained model and modify the final layer to match the number of classes.

    Args:
        model_name (str): Name of the pretrained model to load (default is "resnet18").
        num_classes (int): Number of output classes for the final layer.
    
    Returns:
        nn.Module: The modified pretrained model.
    """
    if model_name == "resnet18":
        model = models.resnet18(pretrained=True)
    elif model_name == "resnet50":
        model = models.resnet50(pretrained=True)
    elif model_name == "resnet101":
        model = models.resnet101(pretrained=True)
    elif model_name == "resnet152":
        model = models.resnet152(pretrained=True)
    elif model_name == "vgg16":
        model = models.vgg16(pretrained=True)
    elif model_name == "vgg19":
        model = models.vgg19(pretrained=True)
    elif model_name == "inception_v3":
        model = models.inception_v3(pretrained=True)
    elif model_name == "alexnet":
        model = models.alexnet(pretrained=True)
    elif model_name == "mobilenet_v2":
        model = models.mobilenet_v2(pretrained=True)
    elif model_name == "densenet121":
        model = models.densenet121(pretrained=True)
    else:
        raise ValueError(f"Model {model_name} not supported.")
    for param in model.parameters():
        param.requires_grad = False  

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model