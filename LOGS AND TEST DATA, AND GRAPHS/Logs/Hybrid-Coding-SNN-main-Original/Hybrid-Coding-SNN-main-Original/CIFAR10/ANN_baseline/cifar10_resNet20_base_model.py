from math import floor

import torch.backends.cudnn as cudnn
import time
import os
import torch
import sys

current_dir = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(current_dir)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # CUDA configuration

# from data.data_loader_cifar10 import build_data
# from data.data_loader_cifar10Edit import build_data

from models.resnet20 import ResNet20
from utils.classification import training, testing
from utils.lib import dump_json, set_seed

# E:\Projects\Python\RP2\Hybrid-Coding-SNN-main\CIFAR10\ANN_baseline
set_seed(1111)

# Load datasets
home_dir = current_dir  # relative path
print(home_dir)
data_dir = home_dir
# data_dir = current_dir.replace("cifar10_vgg16_base_model. py","Dataset/")
ckp_dir = os.path.join(home_dir, 'exp/mnist/')


# batch_size = 128
num_workers = 0
if "poke" in ckp_dir.lower():
    from data.data_loader_custom import build_data
    batch_size = 128
    image_s = 32
    l_features = int((image_s**2)/2)
    train_loader, test_loader, num_class = build_data(dpath=os.path.join(data_dir, 'data/Poke/'), batch_size=batch_size, workers=num_workers,
                                                      cutout=True, use_cifar10=True, auto_aug=True, image_s=image_s)
elif "brain" in ckp_dir.lower():
    from data.data_loader_custom import build_data

    batch_size = 32
    image_s = 56
    l_features = int(((round(image_s / 32) * 32) ** 2) / 2)
    train_loader, test_loader, num_class = build_data(dpath=os.path.join(data_dir, 'data/Brain/'), batch_size=batch_size, workers=num_workers,
                                                      cutout=True, use_cifar10=True, auto_aug=True, image_s=image_s)
elif "dogs" in ckp_dir.lower():
    from data.data_loader_custom import build_data

    batch_size = 32
    image_s = 56
    l_features = int(((round(image_s / 32) * 32) ** 2) / 2)
    train_loader, test_loader, num_class = build_data(dpath=os.path.join(data_dir, 'data/dogs/'), batch_size=batch_size, workers=num_workers,
                                                      cutout=True, use_cifar10=True, auto_aug=True, image_s=image_s)
elif "mnist" in ckp_dir.lower():
    from data.data_loader_MNISTEdit import build_data
    batch_size = 128
    image_s = 32
    #E:\Projects\Python\RP2\Hybrid-Coding-SNN-main\CIFAR10\ANN_baseline\Dataset
    l_features = int((image_s**2)/2)
    train_loader, test_loader, num_class = build_data(dpath=os.path.join(data_dir, 'data/mnist'), batch_size=batch_size, workers=num_workers,
                                                      cutout=True, use_cifar10=True, auto_aug=True, image_s=image_s)
else:
    from data.data_loader_cifar10Edit import build_data
    batch_size = 128
    image_s = 32
    #E:\Projects\Python\RP2\Hybrid-Coding-SNN-main\CIFAR10\ANN_baseline\Dataset
    l_features = int((image_s**2)/2)
    train_loader, test_loader, num_class = build_data(dpath=os.path.join(data_dir, 'CIFAR10/ANN_baseline/Dataset'), batch_size=batch_size, workers=num_workers,
                                                      cutout=True, use_cifar10=True, auto_aug=True, image_s=image_s)
if __name__ == '__main__':
    print(l_features)
    if torch.cuda.is_available():
        device = 'cuda'
        print('GPU is available')
    else:
        device = 'cpu'
        print('GPU is not available')

    # Parameters
    num_epochs = 90
    # num_epochs = 300
    global best_acc
    best_acc = 0
    test_acc_history = []
    train_acc_history = []

    # Models and training configuration
    model = ResNet20(num_class=num_class, l_features=l_features)
    print(model)
    model = model.to(device)
    cudnn.benchmark = True

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=num_epochs)

    for epoch in range(num_epochs):
        since = time.time()

        # Training Stage
        model, acc_train, loss_train = training(model, train_loader, optimizer, criterion, device)

        # Testing Stage
        acc_test, loss_test = testing(model, test_loader, criterion, device)

        scheduler.step()

        # log results
        test_acc_history.append(acc_test)
        train_acc_history.append(acc_train)

        # Training Record
        time_elapsed = time.time() - since
        print('Epoch {:d} takes {:.0f}m {:.0f}s'.format(epoch + 1, time_elapsed // 60, time_elapsed % 60))
        print('Train Accuracy: {:4f}, Loss: {:4f}'.format(acc_train, loss_train))
        print('Test Accuracy: {:4f}'.format(acc_test))

        # Save Model
        if acc_test > best_acc:
            print("Saving the model.")
            if not os.path.isdir(ckp_dir + 'checkpoint'):
                os.makedirs(ckp_dir + 'checkpoint')

            state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_train,
                'acc': acc_test,
            }
            torch.save(state, ckp_dir + 'checkpoint/resnet20_addFC4096_wAvgPool_baseline.pth')
            best_acc = acc_test

    print('Best Test Accuracy: {:4f}'.format(best_acc))
    training_record = {
        'test_acc_history': test_acc_history,
        'train_acc_history': train_acc_history,
        'best_acc': best_acc,
    }
    dump_json(training_record, ckp_dir, 'resnet20_baseline_train_record')
print('end')
