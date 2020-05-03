from enum import Enum
from datetime import datetime
import importlib
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter


from bn import VGGToy, VGG11

class ModelType(str, Enum):
    VGG11 = 'VGG11'
    VGG_Toy = 'VGGToy'


def test_loop(model, testloader, writer, device, loss_fn, n_iter):
    model.eval()
    correct = 0
    total = 0

    running_loss = 0

    with torch.no_grad():
        for data, labels in testloader:
            data = data.to(device)
            labels = labels.to(device)

            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss_fn(outputs, labels)

    accuracy = (correct / total) * 100.
    loss = (running_loss / total)
    writer.add_scalar('testing/loss', loss, n_iter)
    writer.add_scalar('testing/accuracy', accuracy, n_iter)
    print('Accurarcy: %f' % accuracy)

def train_vgg(model_type=ModelType.VGG_Toy, batch_size=4, batch_norm=False, noise=False, learning_rate=0.01, num_epochs=2):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
         ]
    )

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('training on %s' % device)

    module = importlib.import_module("bn")
    class_ = getattr(module, model_type.value)
    model = class_(device=device, num_classes=10, init_weights=True, batch_norm=batch_norm, noise_injection=noise)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    model.to(device)

    model_id = ''
    if batch_norm:
        model_id += 'batch_norm'
    if noise:
        model_id += 'noise'

    writer = SummaryWriter(log_dir='runs/%s_%s_%s' % (model_type.value, model_id, datetime.now().strftime("%H:%M:%S")))

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        print(trainloader)

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            n_iter = (epoch * len(trainloader)) + i

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            writer.add_scalar('training/loss', loss.item(), n_iter)

            for inps_idx, inps in enumerate(model.layer_inputs):
                inps = inps.cpu().numpy()
                writer.add_scalar('inputs/layer%i/mean' % (inps_idx + 1), inps.mean(), n_iter)
                writer.add_histogram('inputs/layer%i/dist' % (inps_idx + 1), inps, n_iter)

            running_loss += loss.item()
            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

                test_loop(model, testloader, writer, device, criterion, n_iter)
