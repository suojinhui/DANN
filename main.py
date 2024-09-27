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

@torch.no_grad()
def test(my_net, dataloader, dataset_name, epoch, model_root, device):
    """
    用于测试模型的准确率
    """
    log_file_path = '{0}/training_log.txt'.format(model_root)

    """ validation """

    my_net = my_net.eval()

    len_dataloader = len(dataloader)
    data_target_iter = iter(dataloader)

    i = 0
    n_total = 0
    n_correct = 0

    while i < len_dataloader:

        # 开始测试
        data_target = next(data_target_iter)
        t_img, t_label = data_target
        
        t_img = t_img.to(device)
        t_label = t_label.to(device)

        class_output, _ = my_net(input_data=t_img, alpha=0)
        pred = class_output.data.max(1, keepdim=True)[1]
        n_correct += pred.eq(t_label.data.view_as(pred)).cpu().sum()
        n_total += len(t_label)
        i += 1

    accu = n_correct.data.numpy() * 1.0 / n_total
    with open(log_file_path, 'a') as log_file:  # 'a'模式表示追加内容
            log_file.write('epoch: {}, accuracy of the {} dataset: {}\n'.format(epoch, dataset_name, accu))

    print('epoch: {}, accuracy of the {} dataset: {}'.format(epoch, dataset_name, accu))
    
    return accu


def main():
    """
    训练脚本
    """

    # 数据集信息设置
    target_dataset_name = 'mnist_m'
    target_image_root = os.path.join('.', 'dataset', target_dataset_name)

    # 创建保存模型的目录
    model_root = os.path.join('./', 'runs/', 'run_' + str(int(time.time())))
    if not os.path.exists(model_root):# 加时间戳防止重名
        os.makedirs(model_root)
        
    # 创建一个文件用于保存训练中的日志    
    log_file_path = '{0}/training_log.txt'.format(model_root)

    # 超参数设置
    cudnn.benchmark = True
    lr = 1e-3
    batch_size = 128
    image_size = 28
    n_epoch = 50

    # 指定训练设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 记录损失值变化
    loss_history = {
        'loss_s_label': [],
        'loss_s_domain': [],
        'loss_t_domain': [],
        'acc_source': [],
        'acc_target': []
    }

    # 训练随机种子设置
    manual_seed = random.randint(1, 10000)
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)

    # 数据预处理定义
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

    # 源域训练数据集
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

    # 目标域训练数据集
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

    # 源域测试数据集
    dataset_source_test = datasets.MNIST(
                root='./dataset',
                train=False,
                transform=img_transform_source,
            )

    dataloader_source_test = torch.utils.data.DataLoader(
            dataset=dataset_source_test,
            batch_size=batch_size,
            shuffle=False,
            num_workers=8
        )

    # 目标域测试数据集
    test_list = os.path.join(target_image_root, 'mnist_m_test_labels.txt')

    dataset_target_test = GetLoader(
        data_root=os.path.join(target_image_root, 'mnist_m_test'),
        data_list=test_list,
        transform=img_transform_target
    )

    dataloader_target_test = torch.utils.data.DataLoader(
            dataset=dataset_target_test,
            batch_size=batch_size,
            shuffle=False,
            num_workers=8
        )

    # 创建模型
    my_net = CNNModel()
    
    # 设置模型参数为可训练
    for p in my_net.parameters():
        p.requires_grad = True

    # 设置优化器
    optimizer = optim.Adam(my_net.parameters(), lr=lr)

    # 定义损失函数
    loss_class = torch.nn.NLLLoss() # 这里损失函数是作为一个nn.Module的子类，所以下面要指定设备，即使里面没有任何参数
    loss_domain = torch.nn.NLLLoss()

    # 移动到GPU上
    my_net = my_net.to(device)
    loss_class = loss_class.to(device)
    loss_domain = loss_domain.to(device)

    # 循环训练
    for epoch in range(n_epoch):
        my_net.train()
        len_dataloader = min(len(dataloader_source), len(dataloader_target))
        data_source_iter = iter(dataloader_source)
        data_target_iter = iter(dataloader_target)

        i = 0
        while i < len_dataloader:

            # 计算一个梯度反转因子，随着训练的进行逐渐减小
            p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            # 使用源域数据训练
            data_source = next(data_source_iter)
            s_img, s_label = data_source
            
            optimizer.zero_grad()
            batch_size = len(s_label)
            
            domain_label = torch.zeros(batch_size)
            domain_label = domain_label.long()

            s_img = s_img.to(device)
            s_label = s_label.to(device)
            domain_label = domain_label.to(device)

            class_output, domain_output = my_net(input_data=s_img, alpha=alpha)
            loss_s_label = loss_class(class_output, s_label)
            loss_history['loss_s_label'].append(loss_s_label.cpu().data.numpy())
            loss_s_domain = loss_domain(domain_output, domain_label)
            loss_history['loss_s_domain'].append(loss_s_domain.cpu().data.numpy())

            # 使用目标域数据训练
            data_target = next(data_target_iter)
            t_img, t_label = data_target
            
            batch_size = len(t_label)

            domain_label = torch.ones(batch_size)
            domain_label = domain_label.long()
            
            t_img = t_img.to(device)
            domain_label = domain_label.to(device)

            _, domain_output = my_net(input_data=t_img, alpha=alpha)
            loss_t_domain = loss_domain(domain_output, domain_label)
            loss_history['loss_t_domain'].append(loss_t_domain.cpu().data.numpy())
            loss = loss_s_label  + loss_t_domain + loss_s_domain
            
            loss.backward()
            optimizer.step()

            i += 1
            if i % 100 == 0:
                with open(log_file_path, 'a') as log_file:  # 'a'模式表示追加内容
                    log_file.write('epoch: {}, [iter: {} / all {}], loss_s_label: {}, loss_s_domain: {}, loss_t_domain: {}\n'.format(
                        epoch, i, len_dataloader, 
                        loss_s_label.cpu().data.numpy(), loss_s_domain.cpu().data.numpy(), loss_t_domain.cpu().data.numpy()))

                print('epoch: {}, [iter: {} / all {}], loss_s_label: {}, loss_s_domain: {}, loss_t_domain: {}'.format(epoch, i, len_dataloader, 
                                                        loss_s_label.cpu().data.numpy(), loss_s_domain.cpu().data.numpy(), loss_t_domain.cpu().data.numpy()))

        # 保存模型
        torch.save(my_net, '{0}/mnist_mnistm_model_epoch.pth'.format(model_root))
        # 开始验证
        acc_source = test(my_net, dataloader_source_test, "minist", epoch, model_root, device)
        loss_history['acc_source'].append(acc_source)
        acc_target = test(my_net, dataloader_target_test, "minist_m", epoch, model_root, device)
        loss_history['acc_target'].append(acc_target)

    # 开始绘图    
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
    
if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print('Running time: %s Seconds'%(end_time-start_time))