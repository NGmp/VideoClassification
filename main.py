import torch
import torchvision
from torch import optim, nn
import  VideoDataset
import Model

batch_size = 1
dataset_train = VideoDataset.VideoDataset()
dataset_val = VideoDataset.VideoDataset(dataset='val')
dataset_test = VideoDataset.VideoDataset(dataset= 'test')
loaders = {'train' :torch.utils.data.DataLoader(dataset_train, batch_size=batch_size ,shuffle=True, num_workers=1), 'val': torch.utils.data.DataLoader(dataset_val,  batch_size=batch_size ,shuffle=False, num_workers=1), 'test': torch.utils.data.DataLoader(dataset_test,  batch_size=batch_size ,shuffle=False, num_workers=1)}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = Model.VideoClassifier(num_classes=3)
model.to(device)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()
phases = ['train','val']
num_epochs = 10
for epoch in range(num_epochs):
    for phase in phases:
        if phase == 'train':
            model.train()
        else:
            model.eval()
        best_val = 0.0
        running_corrects = 0
        total = 0
        count = 0
        for videos, labels in loaders[phase]:
            videos = videos.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            videos = videos.to(device)
            labels = labels.to(device)
            with torch.set_grad_enabled(phase == 'train'):
                output = model(videos)
                loss = criterion(output, labels)
                _, preds = torch.max(output, 1)
                running_corrects += torch.sum(preds == labels.data)

                total += videos.shape[0]
                count += 1
                if count % 10 == 0:
                    print('Running accuracy : {}, phase '.format(running_corrects / total, phase))
                if phase == 'train':
                    loss.backward()

                    optimizer.step()

        epoch_acc = running_corrects/total, phase
        print('Epoch accuracy : {} , phase '.format())
        if phase == 'val':
            if epoch_acc > best_val:
                torch.save(model.state_dict(), 'VideoClassifier.pth')
                best_val = epoch_acc
                print('New best acc : {}'.format(best_val))