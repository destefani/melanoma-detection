import torch
import torchvision


class ResNet(torch.nn.Module):
    def __init__(self,
        resnet_type = "resnet18",
        pretrained = True,
        num_classes = 2,
        ):
        super(ResNet, self).__init__()

        # Initialize resnet model
        if resnet_type == "resnet18":
            self.resnet = torchvision.models.resnet18(pretrained=True)
            for param in self.resnet.parameters():
                param.requires_grad = False
            self.resnet.fc = torch.nn.Linear(512, num_classes)
            for param in self.resnet.fc.parameters():
                param.requires_grad = True

        if resnet_type == "resnet50":
            self.resnet = torchvision.models.resnet50(pretrained=True)
            for param in self.resnet.parameters():
                param.requires_grad = False
            self.resnet.fc = torch.nn.Linear(2048, num_classes)
            for param in self.resnet.fc.parameters():
                param.requires_grad = True
                
        if resnet_type == "resnet101":
            self.resnet = torchvision.models.resnet101(pretrained=True)
            for param in self.resnet.parameters():
                param.requires_grad = False
            self.resnet.fc = torch.nn.Linear(2048, num_classes)
            for param in self.resnet.fc.parameters():
                param.requires_grad = True
    
    def forward(self, x):
        x = self.resnet(x)
        return x