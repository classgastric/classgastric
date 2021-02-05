from __future__ import print_function, division
from param import *
import torch.optim as optim
from torch.optim import lr_scheduler
import glob
from Functions import *
import torchvision
from torchvision import datasets, transforms
from deform_attunet import AttU_Net_deform
import time
import os


data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(img_size, interpolation=3),
        transforms.RandomHorizontalFlip(p=p_HorizontalFlip),
        transforms.RandomVerticalFlip(p=p_VerticalFlip),
        transforms.RandomRotation((-rotate_degree, rotate_degree)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(img_size, interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = 'data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = torchvision.models.resnet50(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model.fc.in_features
num_class = len(glob.glob(data_dir + '/train/*'))
model.fc = nn.Linear(num_ftrs, num_class)

model = model.to(device)

criterion = nn.CrossEntropyLoss()

since = time.time()

deform_model = AttU_Net_deform(img_ch=3, output_ch=2, start_channel=8).to(device)
sample_grid = generate_grid(img_size).to(device)
deform_transform = deform_SpatialTransform()

params_to_update = list(model.fc.parameters()) + list(deform_model.parameters())
optimizer = optim.Adam(params_to_update, lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)

for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)

    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()  # Set model to training mode
            deform_model.train()
        else:
            model.eval()  # Set model to evaluate mode
            deform_model.eval()

        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
                dvf = deform_model(inputs)
                dvf = deformRange * dvf
                inputs = deform_transform(inputs, dvf, sample_grid)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        if phase == 'train':
            scheduler.step()

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            phase, epoch_loss, epoch_acc))