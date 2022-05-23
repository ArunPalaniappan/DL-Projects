import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters 
num_epochs = 5
batch_size = 4
learning_rate = 0.001

# dataset has PILImage images of range [0, 1]. 
# We transform them to Tensors of normalized range [-1, 1]
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# CIFAR10: 60000 32x32 color images in 10 classes, with 6000 images per class
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                          shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                         shuffle=False)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(train_loader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1a = nn.Conv2d(3, 64, 3, padding='same')
        self.conv1b = nn.Conv2d(64, 64, 3, padding='same')
        
        self.conv2a = nn.Conv2d(64, 128, 3, padding='same')
        self.conv2b = nn.Conv2d(128, 128, 3, padding='same')

        self.conv3a = nn.Conv2d(128, 256, 3, padding='same')
        self.conv3b = nn.Conv2d(256, 256, 3, padding='same')
        
        self.conv4a = nn.Conv2d(256, 512, 3, padding='same')
        self.conv4b = nn.Conv2d(512, 512, 3, padding='same')
        
        self.conv5 = nn.Conv2d(512, 512, 3, padding='same')
        
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(512 * 1 * 1, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 10)

    def forward(self, x):

        x = F.relu(self.conv1a(x))
        x = self.pool(F.relu(self.conv1b(x)))
        
        x = F.relu(self.conv2a(x))
        x = self.pool(F.relu(self.conv2b(x)))
        
        x = F.relu(self.conv3a(x))
        x = F.relu(self.conv3b(x))
        x = self.pool(F.relu(self.conv3b(x)))

        x = F.relu(self.conv4a(x))
        x = F.relu(self.conv4b(x))
        x = self.pool(F.relu(self.conv4b(x)))

        x = F.relu(self.conv5(x))
        x = F.relu(self.conv5(x))
        x = self.pool(F.relu(self.conv5(x)))

        x = x.view(-1, 512 * 1 * 1)

        x = F.relu(self.fc1(x))           
        x = F.relu(self.fc2(x))        
        x = self.fc3(x) 
        
        return x


model = ConvNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 2000 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

print('Finished Training')
PATH = './cnn.pth'
torch.save(model.state_dict(), PATH)

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
        
        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')

    for i in range(10):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {acc} %')
