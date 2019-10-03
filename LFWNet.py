# ------------------------------------------------------------------------------------------------------------
# https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py

import torch
import torch.nn as nn # neural network lib.

torch.set_default_tensor_type('torch.cuda.FloatTensor')

class LFWNet(nn.Module):

    def load_state(self, state_dict):
        """
        Load the network from state_dict with the key name, can be used for loading part of pre-trained network
        :param state_dict: state_dict instance, use torch.load() to get the state_dict
        """
        cur_model_state = self.state_dict()
        input_state = {k: v for k, v in state_dict.items() if
                       k in cur_model_state and v.size() == cur_model_state[k].size()}
        cur_model_state.update(input_state)
        self.load_state_dict(cur_model_state)

    def __init__(self):
        super(LFWNet, self).__init__()

        # From the Alexnet
        self.alexnet = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 14),
        )

    def forward(self, x):
        x = self.alexnet(x)
        x = x.view(-1,  6 * 6 * 256)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    model_urls = {
        'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
    }
    model = LFWNet()

    # model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))  # not working at Pycharm
    # torch.load('alexnet-owt-4df8aa71.pth') # not working at Pycharm
    '''
    (1) download :  https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth
    (2) Open the terminal,
      >>cd (the location where the .pth file is)
      >>python
      >>import torch
      >>dict = torch.load('alexnet-owt-4df8aa71.pth')
      >>torch.save(dict, "alexnet_parameters.pth")
    '''
    model.load_state(torch.load('alexnet_parameters.pth'))
    print(model)
    random_input = torch.rand(1, 3, 225, 225)  # (N, C, H, W)
    print(random_input.shape)
    y = model.forward(random_input)
    print(y.shape)
