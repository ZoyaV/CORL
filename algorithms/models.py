
import numpy as np
import torch
from torch import nn


def calc_num_elements(module, module_input_shape):
    shape_with_batch_dim = (1,) + module_input_shape
    some_input = torch.rand(shape_with_batch_dim)
    num_elements = module(some_input).numel()
    return num_elements

    
class ResudualBlock(nn.Module):
    def __init__(self, input_ch, output_ch):
        super().__init__()


        layers = [
            nn.ReLU(),
            nn.Conv2d(input_ch, output_ch, kernel_size=3, stride=1, padding=1),  # padding SAME
            nn.ReLU(),
            nn.Conv2d(output_ch, output_ch, kernel_size=3, stride=1, padding=1),  # padding SAME
        ]

        self.res_block_core = nn.Sequential(*layers)

    def forward(self, x):
        out = self.res_block_core(x)       
        return out

class EncoderBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_after_enc = None
        self.encoder_out_size = -1  # to be initialized in the constuctor of derived class

    def get_encoder_out_size(self):
        return self.encoder_out_size

    def model_to_device(self, device):
        """Default implementation, can be overridden in derived classes."""
        self.to(device)

    def device_and_type_for_input_tensor(self, _):
        """Default implementation, can be overridden in derived classes."""
        return self.model_device(), torch.float32

    def model_device(self):
        return next(self.parameters()).device

    def forward_fc_blocks(self, x):
        if self.fc_after_enc is not None:
            x = self.fc_after_enc(x)

        return x
    
class ResNetEncoder(EncoderBase):
    def __init__(self,observation_shape, feature_size):
        super().__init__()
        self.feature_size = feature_size
        obs_shape = observation_shape
      #  raise Exception(observation_shape)
        input_ch = obs_shape[0]

       # resnet_conf = [[16, 2], [32, 2], [32, 2]]
        resnet_conf = [[64, 3]]
        curr_input_channels = input_ch
        layers = []
        for i, (out_channels, res_blocks) in enumerate(resnet_conf):
            layers.extend([
                nn.Conv2d(curr_input_channels, out_channels, kernel_size=3, stride=1, padding=1),  # padding SAME
              #  nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # padding SAME
            ])

            for j in range(res_blocks):
                layers.append(ResudualBlock(out_channels, out_channels))

            curr_input_channels = out_channels

        layers.append(nn.ReLU())

        self.conv_head = nn.Sequential(*layers)
        self.conv_head_out_size = calc_num_elements(self.conv_head, obs_shape)
        #log.debug('Convolutional layer output size: %r', self.conv_head_out_size)

        self.init_fc_blocks(self.conv_head_out_size)

    def forward(self, obs_dict):
        
       # print(obs_dict)
        x = self.conv_head(obs_dict.reshape(-1, 12, 21, 21))
        x = x.contiguous().view(-1, self.conv_head_out_size)
      #  print()
        x = self.forward_fc_blocks(x)
       # print(x.shape)
        try:
            return x.reshape(64, 20, 128)
        except:
            return x.reshape(1, -1, 128)
    
    def get_feature_size(self):
        return self.feature_size
    
    def init_fc_blocks(self, input_size):
        layers = []
        fc_layer_size = self.feature_size
        encoder_extra_fc_layers = 1
        for i in range(encoder_extra_fc_layers):
            size = input_size if i == 0 else fc_layer_size

            layers.extend([
                nn.Linear(size, fc_layer_size),
                nn.ReLU(),
            ])

        if len(layers) > 0:
            self.fc_after_enc = nn.Sequential(*layers)
            self.encoder_out_size = fc_layer_size
        else:
            self.encoder_out_size = input_size
            
    def get_feature_size(self):
        return self.feature_size
