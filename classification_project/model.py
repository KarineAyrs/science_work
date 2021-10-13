import timm
import torch
import torch.nn as nn
import torch.optim as optim


class TimmModel(nn.Module):

    def __init__(self, model_name='efficientnet_b0', pretrained=True, learning_rate=0.001, batch_size=2, num_epochs=5,
                 criterion=None, optimizer=None):
        super(TimmModel, self).__init__()
        self.model_name = model_name
        self._pretrained = pretrained
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = timm.create_model(model_name=model_name, pretrained=pretrained)
        self.model.train()
        if torch.cuda.is_available():
            self.model.cuda()
        self.lr = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.criterion = nn.CrossEntropyLoss() if criterion is None else criterion
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate) if optimizer is None else optimizer
