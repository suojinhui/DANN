import random
import os
import time
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from dataset.data_loader import GetLoader
from torchvision import datasets
from torchvision import transforms
from models.model import CNNModel
import matplotlib.pyplot as plt
import numpy as np
from test import test

source_dataset_name = 'MNIST'
target_dataset_name = 'mnist_m'
source_image_root = os.path.join('.', 'dataset', source_dataset_name)
target_image_root = os.path.join('.', 'dataset', target_dataset_name)
# 创建保存模型的目录
model_root = os.path.join('./', 'runs/', 'run_' + str(int(time.time())))
if not os.path.exists(model_root):# 加时间戳防止重名
    os.makedirs(model_root)
# 创建一个文件用于保存训练中的日志    
log_file_path = '{0}/training_log.txt'.format(model_root)

# hyper-parameters
cuda = True
cudnn.benchmark = True
lr = 1e-3
batch_size = 128
image_size = 28
n_epoch = 50

# 记录损失值变化
loss_history = {
    'loss_s_label': [],
    'loss_s_domain': [],
    'loss_t_domain': [],
    'acc_source': [],
    'acc_target': []
}

manual_seed = random.randint(1, 10000)
random.seed(manual_seed)
torch.manual_seed(manual_seed)

# load data

img_transform_source = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.1307,), std=(0.3081,))
])

img_transform_target = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

dataset_source = datasets.MNIST(
    root='./dataset',
    train=True,
    transform=img_transform_source,
    download=True
)

dataloader_source = torch.utils.data.DataLoader(
    dataset=dataset_source,
    batch_size=batch_size,
    shuffle=True,
    num_workers=8)

train_list = os.path.join(target_image_root, 'mnist_m_train_labels.txt')

dataset_target = GetLoader(
    data_root=os.path.join(target_image_root, 'mnist_m_train'),
    data_list=train_list,
    transform=img_transform_target
)

dataloader_target = torch.utils.data.DataLoader(
    dataset=dataset_target,
    batch_size=batch_size,
    shuffle=True,
    num_workers=8)

# load model

my_net = CNNModel()

# setup optimizer

optimizer = optim.Adam(my_net.parameters(), lr=lr)

loss_class = torch.nn.NLLLoss()
loss_domain = torch.nn.NLLLoss()

if cuda:
    my_net = my_net.cuda()
    loss_class = loss_class.cuda()
    loss_domain = loss_domain.cuda()

for p in my_net.parameters():
    p.requires_grad = True

# training

for epoch in range(n_epoch):

    len_dataloader = min(len(dataloader_source), len(dataloader_target))
    data_source_iter = iter(dataloader_source)
    data_target_iter = iter(dataloader_target)

    i = 0
    while i < len_dataloader:

        p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        # training model using source data
        data_source = data_source_iter.next()
        s_img, s_label = data_source

        my_net.zero_grad()
        batch_size = len(s_label)

        input_img = torch.FloatTensor(batch_size, 3, image_size, image_size)
        class_label = torch.LongTensor(batch_size)
        domain_label = torch.zeros(batch_size)
        domain_label = domain_label.long()

        if cuda:
            s_img = s_img.cuda()
            s_label = s_label.cuda()
            input_img = input_img.cuda()
            class_label = class_label.cuda()
            domain_label = domain_label.cuda()

        input_img.resize_as_(s_img).copy_(s_img)
        class_label.resize_as_(s_label).copy_(s_label)

        class_output, domain_output = my_net(input_data=input_img, alpha=alpha)
        loss_s_label = loss_class(class_output, class_label)
        loss_history['loss_s_label'].append(loss_s_label.cpu().data.numpy())
        loss_s_domain = loss_domain(domain_output, domain_label)
        loss_history['loss_s_domain'].append(loss_s_domain.cpu().data.numpy())

        # training model using target data
        data_target = data_target_iter.next()
        t_img, _ = data_target

        batch_size = len(t_img)

        input_img = torch.FloatTensor(batch_size, 3, image_size, image_size)
        domain_label = torch.ones(batch_size)
        domain_label = domain_label.long()

        if cuda:
            t_img = t_img.cuda()
            input_img = input_img.cuda()
            domain_label = domain_label.cuda()

        input_img.resize_as_(t_img).copy_(t_img)

        _, domain_output = my_net(input_data=input_img, alpha=alpha)
        loss_t_domain = loss_domain(domain_output, domain_label)
        loss_history['loss_t_domain'].append(loss_t_domain.cpu().data.numpy())
        loss = loss_s_label  + loss_t_domain + loss_s_domain
        loss.backward()
        optimizer.step()

        i += 1
        if i % 100 == 0:
            if domain_output is not None:
                with open(log_file_path, 'a') as log_file:  # 'a'模式表示追加内容
                    log_file.write('epoch: {}, [iter: {} / all {}], loss_s_label: {}, loss_s_domain: {}, loss_t_domain: {}\n'.format(
                        epoch, i, len_dataloader, 
                        loss_s_label.cpu().data.numpy(), loss_s_domain.cpu().data.numpy(), loss_t_domain.cpu().data.numpy()))

                print('epoch: {}, [iter: {} / all {}], loss_s_label: {}, loss_s_domain: {}, loss_t_domain: {}'.format(epoch, i, len_dataloader, 
                                                        loss_s_label.cpu().data.numpy(), loss_s_domain.cpu().data.numpy(), loss_t_domain.cpu().data.numpy()))
            else:
                with open(log_file_path, 'a') as log_file:  # 'a'模式表示追加内容
                    log_file.write('epoch: {}, [iter: {} / all {}], loss_s_label: {}\n'.format(
                        epoch, i, len_dataloader, 
                        loss_s_label.cpu().data.numpy()))

                print('epoch: {}, [iter: {} / all {}], loss_s_label: {}'.format(epoch, i, len_dataloader, 
                                                        loss_s_label.cpu().data.numpy()))


    torch.save(my_net, '{0}/mnist_mnistm_model_epoch.pth'.format(model_root))
    acc_source = test(source_dataset_name, epoch, model_root)
    loss_history['acc_source'].append(acc_source)
    acc_target = test(target_dataset_name, epoch, model_root)
    loss_history['acc_target'].append(acc_target)
    
plt.figure(figsize=(10, 5))
plt.plot(loss_history['loss_s_label'], label='loss_s_label')
if loss_history['loss_s_domain']:
    plt.plot(loss_history['loss_s_domain'], label='loss_s_domain')
if loss_history['loss_t_domain']:
    plt.plot(loss_history['loss_t_domain'], label='loss_t_domain')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss during Training')
plt.savefig('{0}/loss_curve.png'.format(model_root))

plt.figure(figsize=(10, 5))
plt.plot(loss_history['acc_source'], label='acc_source')
plt.plot(loss_history['acc_target'], label='acc_target')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.title('Accuracy during Training')
plt.savefig('{0}/acc_curve.png'.format(model_root))

print('done')