# Session 7 models
#-----------------------------------------------------------------------------------------
import  torch
import  torch.nn as nn
import  torch.nn.functional as F
from torchsummary import summary
from tqdm import tqdm
import matplotlib.pyplot as plt

#Base Model
class Net_1(nn.Module):
    def __init__(self):
        super(Net_1, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, 3, padding=1) # 28>28 | 3
        self.conv2 = nn.Conv2d(4, 4, 3, padding=1) # 28 > 28 |  5
        self.pool1 = nn.MaxPool2d(2, 2) # 28 > 14 | 10
        self.conv3 = nn.Conv2d(4, 8, 3, padding=1) # 14> 14 | 12
        self.conv4 = nn.Conv2d(8, 8, 3, padding=1) #14 > 14 | 14
        self.pool2 = nn.MaxPool2d(2, 2) # 14 > 7 | 28
        self.conv5 = nn.Conv2d(8, 8, 3) # 7 > 5 | 30
        self.conv6 = nn.Conv2d(8, 4, 3) # 5 > 3 | 32 | 3*3*4 | 3x3x4x10 | 
        self.conv7 = nn.Conv2d(4, 10, 3) # 3 > 1 | 34 | > 1x1x10
        
    def forward(self, x):
        x = self.pool1(F.relu(self.conv2(F.relu(self.conv1(x)))))
        x = self.pool2(F.relu(self.conv4(F.relu(self.conv3(x)))))
        x = F.relu(self.conv6(F.relu(self.conv5(x))))
        # x = F.relu(self.conv7(x))
        x = self.conv7(x)
        x = x.view(-1, 10) #1x1x10> 10
        return F.log_softmax(x, dim=-1)
        
    def model_summary(model, input_size):
        summary(model, input_size)
        
#model 2
class Net_2(nn.Module):
    def __init__(self):
        super(Net_2, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1) # 28>28 | 3
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1) # 28 > 28 |  5
        self.pool1 = nn.MaxPool2d(2, 2) # 28 > 14 | 10
        self.conv3 = nn.Conv2d(16, 8, 3, padding=1) # 14> 14 | 12
        self.conv4 = nn.Conv2d(8, 8, 3, padding=1) #14 > 14 | 14
        self.pool2 = nn.MaxPool2d(2, 2) # 14 > 7 | 28
        self.conv5 = nn.Conv2d(8, 16, 3) # 7 > 5 | 30
        self.conv6 = nn.Conv2d(16, 8, 3) # 5 > 3 | 32 | 3*3*8 | 3x3x8x10 | 
        self.conv7 = nn.Conv2d(8, 10, 3) # 3 > 1 | 34 | > 1x1x10

    def forward(self, x):
        x = self.pool1(F.relu(self.conv2(F.relu(self.conv1(x)))))
        x = self.pool2(F.relu(self.conv4(F.relu(self.conv3(x)))))
        x = F.relu(self.conv6(F.relu(self.conv5(x))))
        # x = F.relu(self.conv7(x))
        x = self.conv7(x)
        x = x.view(-1, 10) #1x1x10> 10
        return F.log_softmax(x, dim=-1)

    def model_summary(model, input_size):
        summary(model, input_size)

#model 3
class Net_3(nn.Module):
    def __init__(self):
        super(Net_3, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1) # 28>28 | 3
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1) # 28 > 28 |  5
        self.pool1 = nn.MaxPool2d(2, 2) # 28 > 14 | 10
        self.conv3 = nn.Conv2d(16, 8, 3, padding=1) # 14> 14 | 12
        self.conv4 = nn.Conv2d(8, 8, 3, padding=1) #14 > 14 | 14
        self.pool2 = nn.MaxPool2d(2, 2) # 14 > 7 | 28
        self.conv5 = nn.Conv2d(8, 16, 3) # 7 > 5 | 30
        self.conv6 = nn.Conv2d(16, 8, 3) # 5 > 3 | 32 | 3*3*8 | 3x3x8x10 | 
        self.conv7 = nn.Conv2d(8, 10, 3) # 3 > 1 | 34 | > 1x1x10
        self.dropout = nn.Dropout(0.5)
    def forward(self, x):
        x = self.pool1(self.dropout(F.relu(self.conv2(F.relu(self.conv1(x))))))
        x = self.pool2(F.relu(self.conv4(F.relu(self.conv3(x)))))
        x = self.dropout(x)
        x = F.relu(self.conv6(F.relu(self.conv5(x))))
        # x = F.relu(self.conv7(x))
        x = self.conv7(x)
        x = x.view(-1, 10) #1x1x10> 10
        return F.log_softmax(x, dim=-1)

    def model_summary(model, input_size):
        summary(model, input_size)

#model 4
class Net_4(nn.Module):
    def __init__(self):
        super(Net_4, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1) # 28>28 | 3
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1) # 28 > 28 |  5
        self.pool1 = nn.MaxPool2d(2, 2) # 28 > 14 | 10
        self.conv3 = nn.Conv2d(16, 8, 3, padding=1) # 14> 14 | 12
        self.conv4 = nn.Conv2d(8, 8, 3, padding=1) #14 > 14 | 14
        self.pool2 = nn.MaxPool2d(2, 2) # 14 > 7 | 28
        self.conv5 = nn.Conv2d(8, 8, 3) # 7 > 5 | 30
        self.conv6 = nn.Conv2d(8, 16, 3) # 5 > 3 | 32 | 3*3*32 | 3x3x32x10 | 
        self.conv7 = nn.Conv2d(16, 10, 3) # 3 > 1 | 34 | > 1x1x10
        self.dropout = nn.Dropout(0.1)
    def forward(self, x):
        x = self.pool1(self.dropout(F.relu(self.conv2(F.relu(self.conv1(x))))))
        x = self.pool2(self.dropout(F.relu(self.conv4(self.dropout(F.relu(self.conv3(x)))))))
        x = self.dropout(F.relu(self.conv6(self.dropout(F.relu(self.conv5(x))))))
        # x = F.relu(self.conv7(x))
        x = self.conv7(x)
        x = x.view(-1, 10) #1x1x10> 10
        return F.log_softmax(x, dim=-1)

    def model_summary(model, input_size):
        summary(model, input_size)

#model 5
class Net_5(nn.Module):
    def __init__(self):
        super(Net_5, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1) # 28>28 | 3
        self.batch1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1) # 28 > 28 |  5
        self.batch2 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2) # 28 > 14 | 10
        self.conv3 = nn.Conv2d(16, 8, 3, padding=1) # 14> 14 | 12
        self.batch3 = nn.BatchNorm2d(8)
        self.conv4 = nn.Conv2d(8, 8, 3, padding=1) #14 > 14 | 14
        self.batch4 = nn.BatchNorm2d(8)
        self.pool2 = nn.MaxPool2d(2, 2) # 14 > 7 | 28
        self.conv5 = nn.Conv2d(8, 8, 3) # 7 > 5 | 30
        self.batch5 = nn.BatchNorm2d(8)
        self.conv6 = nn.Conv2d(8, 16, 3) # 5 > 3 | 32 | 3*3*32 | 3x3x32x10 |
        self.batch6 = nn.BatchNorm2d(16)
        self.conv7 = nn.Conv2d(16, 10, 3) # 3 > 1 | 34 | > 1x1x10
        self.dropout = nn.Dropout(0.1)
    def forward(self, x):
        x = self.pool1(self.dropout(self.batch2(F.relu(self.conv2(self.batch1(F.relu(self.conv1(x))))))))
        x = self.pool2(self.dropout(self.batch4(F.relu(self.conv4(self.dropout(self.batch3(F.relu(self.conv3(x)))))))))
        x = self.dropout(self.batch6(F.relu(self.conv6(self.dropout(self.batch5(F.relu(self.conv5(x))))))))
        # x = F.relu(self.conv7(x))
        x = self.conv7(x)
        x = x.view(-1, 10) #1x1x10> 10
        return F.log_softmax(x, dim=-1)

    def model_summary(model, input_size):
        summary(model, input_size)

#model 6 & 7
class Net_6(nn.Module):
    def __init__(self):
        super(Net_6, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1) # 28>28 | 3
        self.batch1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1) # 28 > 28 |  5
        self.batch2 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2) # 28 > 14 | 10
        self.conv3 = nn.Conv2d(16, 8, 3, padding=1) # 14> 14 | 12
        self.batch3 = nn.BatchNorm2d(8)
        self.conv4 = nn.Conv2d(8, 8, 3, padding=1) #14 > 14 | 14
        self.batch4 = nn.BatchNorm2d(8)
        self.pool2 = nn.MaxPool2d(2, 2) # 14 > 7 | 28
        self.conv5 = nn.Conv2d(8, 8, 3) # 7 > 5 | 30
        self.batch5 = nn.BatchNorm2d(8)
        self.conv6 = nn.Conv2d(8, 16, 3) # 5 > 3 | 32 | 3*3*32 | 3x3x32x10 |
        self.batch6 = nn.BatchNorm2d(16)
        self.conv7 = nn.Conv2d(16, 10, 3) # 3 > 1 | 34 | > 1x1x10
        self.dropout = nn.Dropout(0.1)
    def forward(self, x):
        x = self.pool1(self.dropout(self.batch2(F.relu(self.conv2(self.batch1(F.relu(self.conv1(x))))))))
        x = self.pool2(self.dropout(self.batch4(F.relu(self.conv4(self.dropout(self.batch3(F.relu(self.conv3(x)))))))))
        x = self.dropout(self.batch6(F.relu(self.conv6(self.dropout(self.batch5(F.relu(self.conv5(x))))))))
        # x = F.relu(self.conv7(x))
        x = self.conv7(x)
        x = x.view(-1, 10) #1x1x10> 10
        return F.log_softmax(x, dim=-1)

    def model_summary(model, input_size):
        summary(model, input_size)

#model 8
class Net_8(nn.Module):
    def __init__(self):
        super(Net_8, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1) # 28>28 | 3
        self.batch1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 8, 3, padding=1) # 28 > 28 |  5
        self.batch2 = nn.BatchNorm2d(8)
        self.pool1 = nn.MaxPool2d(2, 2) # 28 > 14 | 10
        self.conv3 = nn.Conv2d(8, 8, 3, padding=1) # 14> 14 | 12
        self.batch3 = nn.BatchNorm2d(8)
        self.conv4 = nn.Conv2d(8, 8, 3, padding=1) #14 > 14 | 14
        self.batch4 = nn.BatchNorm2d(8)
        self.pool2 = nn.MaxPool2d(2, 2) # 14 > 7 | 28
        self.conv5 = nn.Conv2d(8, 8, 3) # 7 > 5 | 30
        self.batch5 = nn.BatchNorm2d(8)
        self.conv6 = nn.Conv2d(8, 16, 3) # 5 > 3 | 32 | 3*3*32 | 3x3x32x10 |
        self.batch6 = nn.BatchNorm2d(16)
        self.conv7 = nn.Conv2d(16, 10, 3) # 3 > 1 | 34 | > 1x1x10
        self.dropout = nn.Dropout(0.1)
    def forward(self, x):
        x = self.pool1(self.dropout(self.batch2(F.relu(self.conv2(self.batch1(F.relu(self.conv1(x))))))))
        x = self.pool2(self.dropout(self.batch4(F.relu(self.conv4(self.dropout(self.batch3(F.relu(self.conv3(x)))))))))
        x = self.dropout(self.batch6(F.relu(self.conv6(self.dropout(self.batch5(F.relu(self.conv5(x))))))))
        # x = F.relu(self.conv7(x))
        x = self.conv7(x)
        x = x.view(-1, 10) #1x1x10> 10
        return F.log_softmax(x, dim=-1)

    def model_summary(model, input_size):
        summary(model, input_size)
#---------------------------------------------------------------------------------------------------------------------------------------
# Session 8 Models
#---------------------------------------------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------------------------------------------
#Train and Test

train_losses = []
test_losses = []
train_acc = []
test_acc = []

def model_train(model, device, train_loader, optimizer, epoch):
  model.train()
  pbar = tqdm(train_loader)
  correct = 0
  processed = 0
  for batch_idx, (data, target) in enumerate(pbar):
    # get samples
    data, target = data.to(device), target.to(device)

    # Init
    optimizer.zero_grad()
    # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes. 
    # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

    # Predict
    y_pred = model(data)

    # Calculate loss
    loss = F.nll_loss(y_pred, target)
    train_losses.append(loss)

    # Backpropagation
    loss.backward()
    optimizer.step()

    # Update pbar-tqdm
    
    pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct += pred.eq(target.view_as(pred)).sum().item()
    processed += len(data)

    pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
    train_acc.append(100*correct/processed)

def model_test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    test_acc.append(100. * correct / len(test_loader.dataset))

def draw_graph():
    t = [t_items.item() for t_items in train_losses]
    fig, axs = plt.subplots(2,2,figsize=(15,10))
    axs[0, 0].plot(t)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")
