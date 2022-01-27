
from torchvision import  models
from torch import optim
from torch import nn

class Model:
    def __init__(self,output_layers,device = 'cpu',lr = 0.003):
        self.device = device
        self.output_layers = output_layers
        self.load_model()
        self.optimizer = optim.Adam(self.model.fc.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min',patience=2,)
        self.criterion = nn.NLLLoss()

    
    
    def load_model(self):
        ##load model
        
        self.model = models.resnet18(pretrained=True)
        
        # print(self.model)s
        # Freeze parameters so we don't backprop through them
        for param in self.model.parameters():
            param.requires_grad = False
            
        self.model.fc = nn.Sequential(nn.Linear(512, 64),
                                        nn.ReLU(),
                                        nn.Dropout(0.2),
                                        nn.Linear(64, self.output_layers),
                                        nn.LogSoftmax(dim=1))
                                        
        self.model.to(self.device)
        
        
    
        
    
