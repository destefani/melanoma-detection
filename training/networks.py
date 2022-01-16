import torch
import torchvision


class ResNet(torch.nn.Module):
    def __init__(self,
        resnet_type = "resnet18",
        pretrained = True,
        num_classes = 1,
        ):
        super(ResNet, self).__init__()

        # Initialize resnet model
        if resnet_type == "resnet18":
            self.resnet = torchvision.models.resnet18(pretrained=pretrained)
            for param in self.resnet.parameters():
                param.requires_grad = False
            self.resnet.fc = torch.nn.Linear(512, num_classes)
            for param in self.resnet.fc.parameters():
                param.requires_grad = True

        if resnet_type == "resnet50":
            self.resnet = torchvision.models.resnet50(pretrained=pretrained)
            for param in self.resnet.parameters():
                param.requires_grad = False
            self.resnet.fc = torch.nn.Linear(2048, num_classes)
            for param in self.resnet.fc.parameters():
                param.requires_grad = True
                
        if resnet_type == "resnet101":
            self.resnet = torchvision.models.resnet101(pretrained=pretrained)
            for param in self.resnet.parameters():
                param.requires_grad = False
            self.resnet.fc = torch.nn.Linear(2048, num_classes)
            for param in self.resnet.fc.parameters():
                param.requires_grad = True

        if resnet_type == "resnet152":
            self.resnet = torchvision.models.resnet152(pretrained=pretrained)
            for param in self.resnet.parameters():
                param.requires_grad = False
            self.resnet.fc = torch.nn.Linear(2048, num_classes)
            for param in self.resnet.fc.parameters():
                param.requires_grad = True
    
    def forward(self, x):
        x = self.resnet(x)
        return x

class EfficientNet7(torch.nn.Module):
    def __init__(self,
        pretrained = True,
        num_classes = 1,
        ):
        super(EfficientNet7, self).__init__()

        # Initialize resnet model
        self.model = torchvision.models.efficientnet_b7(pretrained=pretrained)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.classifier[1] = torch.nn.Linear(2560, num_classes)
        for param in self.model.classifier[1].parameters():
            param.requires_grad = True
    
    def forward(self, x):
        x = self.model(x)
        return x