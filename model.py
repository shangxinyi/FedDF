from torch.nn import Module, Conv2d, Linear, MaxPool2d
from torch.nn import functional as F

#input.shape = [batch size,channel,height,width]
#input_shape = [channel,height,width]
class CifarCnn(Module):
    def __init__(self, input_shape, out_dim):
        super(CifarCnn, self).__init__()
        self.conv1 = Conv2d(input_shape[0], 32, 5)
        self.conv2 = Conv2d(32, 64, 5)
        self.fc1 = Linear(64 * 5 * 5, 512)
        self.fc2 = Linear(512, 128)
        self.fc3 = Linear(128, out_dim)

        def forward(self, x):
            out = F.relu(self.conv1(x))
            out = F.max_pool2d(out, 2)
            out = F.relu(self.conv2(out))
            out = F.max_pool2d(out, 2)
            out = out.view(out.size(0), -1)
            out = F.relu(self.fc1(out))
            out = F.relu(self.fc2(out))
            out = self.fc3(out)
            return out