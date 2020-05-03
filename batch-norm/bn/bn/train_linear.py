from enum import Enum
from datetime import datetime
import importlib

import torch

import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

import torch.nn as nn
import torch.optim as optim


class ModelType(str, Enum):
    LINEAR = 'Linear'
    LINEAR_BN = 'LinearBN'
    LINEAR_BN_NOISE = 'LinearBNNoise'


def train(model_type=ModelType.LINEAR, batch_size=128, num_epochs=2, learning_rate=0.1):
    trainset = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset = torchvision.datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )
    testLoader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('training on %s' % device)

    module = importlib.import_module("bn")
    class_ = getattr(module, model_type.value)
    model = class_(device)

    model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    opt = optim.SGD(model.parameters(), lr=learning_rate)

    loss_arr = []

    writer = SummaryWriter(log_dir='runs/%s_%s' % (model_type.value, datetime.now().strftime("%H:%M:%S")))

    for epoch in range(num_epochs):

        for i, data in enumerate(trainloader, 0):
            model.train()
            n_iter = (epoch * len(trainloader)) + i

            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            opt.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            opt.step()

            loss_arr.append(loss.item())

            writer.add_scalar('training/loss', loss.item(), n_iter)
            writer.add_scalar('inputs/layer1/mean', model.l1_inp.cpu().numpy().mean(), n_iter)
            writer.add_scalar('inputs/layer2/mean', model.l2_inp.cpu().numpy().mean(), n_iter)
            writer.add_histogram('inputs/layer1/dist', model.l1_inp.cpu().numpy(), n_iter)
            writer.add_histogram('inputs/layer2/dist', model.l2_inp.cpu().numpy(), n_iter)

            if i % 10 == 0:
                inputs = inputs.view(inputs.size(0), -1)

                model.eval()
                print('training loss: %0.2f' % loss.item())

                model.eval()
                test_loss = 0
                correct = 0
                with torch.no_grad():
                    for test_data, test_target in testLoader:
                        test_data = test_data.to(device)
                        test_target = test_target.to(device)
                        output = model(test_data)
                        test_loss += loss_fn(output, test_target)
                        pred = output.argmax(dim=1, keepdim=True)
                        correct += pred.eq(test_target.view_as(pred)).sum().item()

                test_loss /= len(testLoader.dataset)

                writer.add_scalar('testing/loss', test_loss, n_iter)
                writer.add_scalar('testing/accuracy', correct/len(testLoader.dataset) * 100., n_iter)



    # compute summary
    l1_mean = [x[0].cpu() for x in model.l1_dist]
    l1_std = [x[1].cpu() for x in model.l1_dist]
    l2_mean = [x[0].cpu() for x in model.l2_dist]
    l2_std = [x[1].cpu() for x in model.l2_dist]

    return l1_mean, l1_std, l2_mean, l2_std, loss_arr, model_type.value
