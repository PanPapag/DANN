import torch.nn as nn
from torch.autograd import Function

class DANN(nn.Module):

    def __init__(self):
        # Construct nn.Module superclass from the derived classs DANNet
        super(DANN, self).__init__()
        # Construct DANNet architecture
        self.feature_extractor = nn.Sequential()
        self.feature_extractor.add_module('f_conv1', nn.Conv2d(3, 64, kernel_size=5))
        self.feature_extractor.add_module('f_bn1', nn.BatchNorm2d(64))
        self.feature_extractor.add_module('f_pool1', nn.MaxPool2d(2))
        self.feature_extractor.add_module('f_relu1', nn.ReLU(True))
        self.feature_extractor.add_module('f_conv2', nn.Conv2d(64, 50, kernel_size=5))
        self.feature_extractor.add_module('f_bn2', nn.BatchNorm2d(50))
        self.feature_extractor.add_module('f_drop1', nn.Dropout2d())
        self.feature_extractor.add_module('f_pool2', nn.MaxPool2d(2))
        self.feature_extractor.add_module('f_relu2', nn.ReLU(True))

        self.label_predictor = nn.Sequential()
        self.label_predictor.add_module('c_fc1', nn.Linear(50 * 4 * 4, 100))
        self.label_predictor.add_module('c_bn1', nn.BatchNorm1d(100))
        self.label_predictor.add_module('c_relu1', nn.ReLU(True))
        self.label_predictor.add_module('c_drop1', nn.Dropout2d())
        self.label_predictor.add_module('c_fc2', nn.Linear(100, 100))
        self.label_predictor.add_module('c_bn2', nn.BatchNorm1d(100))
        self.label_predictor.add_module('c_relu2', nn.ReLU(True))
        self.label_predictor.add_module('c_fc3', nn.Linear(100, 10))
        self.label_predictor.add_module('c_softmax', nn.LogSoftmax())

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(50 * 4 * 4, 100))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

    def forward(self, input, lamda):
        input = input.expand(input.data.shape[0], 3, 28, 28)
        feature = self.feature_extractor(input)
        feature = feature.view(-1, 50 * 4 * 4)
        reverse_feature = ReverseLayer.apply(feature, lamda)
        class_prediction = self.class_classifier(feature)
        domain_prediction = self.domain_classifier(reverse_feature)
        return class_prediction, domain_prediction


def ReverseLayer(Function):

    @staticmethod
    def forward(ctx, x, lamda):
        ctx.lamda = lamda
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.lamda
        return output, None
