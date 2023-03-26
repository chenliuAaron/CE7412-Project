import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import torch
from utils import Progbar
from model import VRNN
from loss_function import loss as Loss
from config import Config
from torch.utils.data import TensorDataset, DataLoader
import pickle


def load_dataset(batch_size):
    train_dataset = datasets.MNIST(root='../data/',
                                   train=True,
                                   transform=transforms.ToTensor(),
                                   download=True)

    test_dataset = datasets.MNIST(root='../data/',
                                  train=False,
                                  transform=transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)
    return train_loader, test_loader


def train(filename, conf):
    with open(filename, 'rb') as file:
        DNAs = pickle.load(file)
        max_len = max([len(vec) for vec in DNAs])
        datasets = np.zeros((len(DNAs), max_len))
        for i in range(len(DNAs)):
            vec = [0] * (max_len - len(DNAs[i])) + DNAs[i]
            datasets[i, :] = np.array(vec)

    tensor_train = torch.Tensor(datasets[0: int(len(datasets) * 0.1), int(max_len * 0.95):])
    tensor_test = torch.Tensor(datasets[int(len(datasets) * 0.1):, int(max_len * 0.95):])
    train_dataset = TensorDataset(tensor_train)
    test_dataset = TensorDataset(tensor_test)
    train_loader = DataLoader(dataset=train_dataset, batch_size=512, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=512, shuffle=True)
    net = VRNN(conf.x_dim, conf.h_dim, conf.z_dim)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.manual_seed_all(112858)
    net.to(device)
    # net = torch.nn.DataParallel(net, device_ids=[0, 1])
    if conf.restore:
        net.load_state_dict(torch.load(conf.checkpoint_path,
                                       map_location="cuda:0" if torch.cuda.is_available() else "cpu"))
        print('Restore model from ' + conf.checkpoint_path)
    optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=0)
    for ep in range(1, conf.train_epoch + 1):
        prog = Progbar(target=4)
        print("At epoch:{}".format(str(ep)))
        for i, data in enumerate(train_loader):
            print('The %d-th batch' % i)
            data = torch.tensor(np.array([item.cpu().detach().numpy() for item in data])).to(torch.int)
            data = data.to(device)
            data = data.squeeze(0)
            package = net(data)
            loss = Loss(package, net.x_emb(data))
            net.zero_grad()
            loss.backward()
            _ = torch.nn.utils.clip_grad_norm_(net.parameters(), 5)
            optimizer.step()
            prog.update(i, exact=[("Training Loss", loss.item())])

        with torch.no_grad():
            x_decoded = net.sampling(conf.x_dim, device)
            x_decoded = x_decoded.cpu().numpy()
            digit = x_decoded.reshape(conf.x_dim, conf.h_dim)

        if ep % conf.save_every == 0:
            torch.save(net.state_dict(), '../checkpoint/Epoch_' + str(ep + 1) + '.pth')


def generating(sample_size, conf):

    net = VRNN(conf.x_dim, conf.h_dim, conf.z_dim)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    # net = torch.nn.DataParallel(net, device_ids=conf.device_ids)
    net.load_state_dict(torch.load(conf.checkpoint_path, map_location="cuda:0" if torch.cuda.is_available() else "cpu"))
    print('Restore model from ' + conf.checkpoint_path)

    with torch.no_grad():
        x_decoded = net.sampling(sample_size, device)
        x_decoded = x_decoded.cpu().numpy()
        # print(x_decoded[0])

    return x_decoded


if __name__ == '__main__':

    conf = Config()
    train('mers-sequences.pkl', conf)
    samples = generating(28, conf)
