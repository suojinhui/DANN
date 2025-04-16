import argparse
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
from tqdm import tqdm

@torch.no_grad()
def test(my_net, dataloader, dataset_name, epoch, model_root, device):
    """测试模型准确率"""
    log_file_path = os.path.join(model_root, 'training_log.txt')
    my_net.eval()
    
    total_correct = 0
    total_samples = 0
    
    for t_img, t_label in dataloader:
        t_img = t_img.to(device)
        t_label = t_label.to(device)
        
        class_output, _ = my_net(t_img, alpha=0)
        pred = class_output.argmax(dim=1, keepdim=True)
        total_correct += pred.eq(t_label.view_as(pred)).sum().item()
        total_samples += len(t_label)
    
    accuracy = total_correct / total_samples
    with open(log_file_path, 'a') as f:
        f.write(f'epoch: {epoch}, {dataset_name} accuracy: {accuracy:.4f}\n')
    
    print(f'epoch: {epoch}, {dataset_name} accuracy: {accuracy:.4f}')
    return accuracy

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Domain adaptation training script')
    parser.add_argument('--target_dataset', type=str, default='mnist_m', 
                      help='Name of the target dataset (default: mnist_m)')
    parser.add_argument('--data_root', type=str, default='./dataset',
                      help='Root directory of the dataset (default: ./dataset)')
    parser.add_argument('--model_dir', type=str, default='./runs',
                      help='Directory to save the model (default: ./runs)')
    parser.add_argument('--lr', type=float, default=1e-3,
                      help='Learning rate (default: 0.001)')
    parser.add_argument('--batch_size', type=int, default=128,
                      help='Batch size (default: 128)')
    parser.add_argument('--image_size', type=int, default=28,
                      help='Input image size (default: 28)')
    parser.add_argument('--epochs', type=int, default=50,
                      help='Number of training epochs (default: 50)')
    parser.add_argument('--seed', type=int, default=None,
                      help='Random seed (default: random)')
    parser.add_argument('--num_workers', type=int, default=4,
                      help='Number of data loading threads (default: 4)')
    parser.add_argument('--amp', action='store_true',
                      help='Enable mixed precision training')
    parser.add_argument('--no_timestamp', action='store_true',
                      help='Do not add a timestamp to the model directory')
    parser.add_argument('--sgd', action='store_true',
                      help='Using SGD optimizer')

    args = parser.parse_args()

    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # 随机种子设置
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:
        cudnn.benchmark = True

    # 创建模型目录
    timestamp = '' if args.no_timestamp else f'_{int(time.time())}'
    model_root = os.path.join(args.model_dir, f'run{timestamp}')
    os.makedirs(model_root, exist_ok=True)

    # 初始化数据结构
    metrics = {
        'loss_s_label': [],
        'loss_s_domain': [],
        'loss_t_domain': [],
        'acc_source': [],
        'acc_target': []
    }

    # 数据预处理
    source_transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    ])

    target_transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    # 数据加载
    def load_data():
        # 源域数据
        source_train = datasets.MNIST(
            root=args.data_root,
            train=True,
            transform=source_transform,
            download=True
        )
        source_test = datasets.MNIST(
            root=args.data_root,
            train=False,
            transform=source_transform
        )

        # 目标域数据
        target_path = os.path.join(args.data_root, args.target_dataset)
        target_train = GetLoader(
            data_root=os.path.join(target_path, 'mnist_m_train'),
            data_list=os.path.join(target_path, 'mnist_m_train_labels.txt'),
            transform=target_transform
        )
        target_test = GetLoader(
            data_root=os.path.join(target_path, 'mnist_m_test'),
            data_list=os.path.join(target_path, 'mnist_m_test_labels.txt'),
            transform=target_transform
        )

        return (
            torch.utils.data.DataLoader(source_train, args.batch_size, shuffle=True, 
                                      num_workers=args.num_workers),
            torch.utils.data.DataLoader(target_train, args.batch_size, shuffle=True,
                                      num_workers=args.num_workers),
            torch.utils.data.DataLoader(source_test, args.batch_size, shuffle=False,
                                      num_workers=args.num_workers),
            torch.utils.data.DataLoader(target_test, args.batch_size, shuffle=False,
                                      num_workers=args.num_workers)
        )

    train_source, train_target, test_source, test_target = load_data()

    # 模型初始化
    model = CNNModel().to(device)
    if args.sgd:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.NLLLoss().to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    best_acc = 0

    # 训练循环
    for epoch in range(args.epochs):
        model.train()
        start_time = time.time()

        pbar = tqdm(zip(train_source, train_target), 
                total=min(len(train_source), len(train_target)),
                desc=f'Epoch {epoch+1}/{args.epochs}',
                ncols=100,  # 进度条宽度
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
        
        for batch_idx, ((src_imgs, src_labels), (tgt_imgs, _)) in enumerate(pbar):
            # 数据准备
            src_imgs = src_imgs.to(device)
            src_labels = src_labels.to(device)
            tgt_imgs = tgt_imgs.to(device)
            
            # 动态alpha计算
            p = (epoch + 1) / args.epochs
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            # 混合精度训练上下文
            with torch.amp.autocast("cuda", enabled=args.amp):
                # 源域处理
                class_out, domain_out = model(src_imgs, alpha)
                loss_class = criterion(class_out, src_labels)
                loss_domain_src = criterion(domain_out, torch.zeros(len(src_imgs)).long().to(device))

                # 目标域处理
                _, domain_out = model(tgt_imgs, alpha)
                loss_domain_tgt = criterion(domain_out, torch.ones(len(tgt_imgs)).long().to(device))

                # 总损失
                total_loss = loss_class + loss_domain_src + loss_domain_tgt

            # 反向传播
            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # 记录指标
            metrics['loss_s_label'].append(loss_class.item())
            metrics['loss_s_domain'].append(loss_domain_src.item())
            metrics['loss_t_domain'].append(loss_domain_tgt.item())

        # 验证和保存
        epoch_time = time.time() - start_time
        metrics['acc_source'].append(test(model, test_source, "source", epoch, model_root, device))
        metrics['acc_target'].append(test(model, test_target, "target", epoch, model_root, device))
        
        if metrics['acc_target'][-1] > best_acc:
            best_acc = metrics['acc_target'][-1]
            # 保存检查点
            checkpoint = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'metrics': metrics
            }
            torch.save(checkpoint, os.path.join(model_root, f'model_best.pth'))
        
        print(f'Epoch {epoch} complete, time: {epoch_time:.2f}s')

    # 结果可视化
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(metrics['loss_s_label'], label='classification loss')
    plt.plot(metrics['loss_s_domain'], label='source domain loss')
    plt.plot(metrics['loss_t_domain'], label='target domain loss')
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(metrics['acc_source'], label='source accuracy')
    plt.plot(metrics['acc_target'], label='target accuracy')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend()

    plt.savefig(os.path.join(model_root, 'training_metrics.png'))
    plt.close()

if __name__ == '__main__':
    start = time.time()
    main()
    print(f'total time: {time.time()-start:.2f} seconds')