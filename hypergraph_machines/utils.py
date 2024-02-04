import os
import shutil
from datetime import datetime
import torch
import torch.nn.functional as F
import numpy as np
import networkx as nx
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from hypergraph_machines.losses import  hg_classification_loss

def l2_norm(param):
    size_reg = param.numel()
    return torch.sqrt(torch.sum(torch.pow(param, 2)) / size_reg)


class BestModelSaver:
    def __init__(self, path):
        self.loss = 1e+6
        self.check_path(path)
        self.path = os.path.join(path, 'checkpoint.pth.tar')
        self.best_path = os.path.join(path, 'model_best.pth.tar')

    @staticmethod
    def check_path(path):
        if not os.path.isdir(path):
            os.makedirs(path)

    def is_best(self, new_loss):
        return self.loss > new_loss

    def save(self, model, optimizer, epoch, loss, acc):
        state = {
            'epoch': epoch,
            'loss': loss,
            'state_dict': model.state_dict(),
            'best_acc1': acc,
            'optimizer' : optimizer.state_dict(),
        }
        torch.save(state, self.path)
        if self.is_best(loss):
            self.loss = loss
            shutil.copyfile(self.path, self.best_path)


def train(model, device, train_loader, optimizer, epoch):
    steps, correct, n_total = 150, 0, 0
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss, d_l = hg_classification_loss(output, target, model)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        n_total += data.shape[0]
        loss.backward()
        optimizer.step()
        if batch_idx % steps == 0 and batch_idx != 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc {:.3f}%'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(),
                100. * correct / n_total))
            print("main loss: {:.3f}, ret loss: {:.3f}".format(d_l["log_likelihood"],
                                                               d_l["regularization"]))


def test(model, device, test_loader, loss_func = F.nll_loss):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_func(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_acc = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), test_acc))
    return test_loss, test_acc


def generate_timestamp():
    return datetime.now().isoformat()[:-7].replace("T","-").replace(":","-")


def get_spaced_colors(n, norm = False, black = True, cmap = 'jet'):
    rgb_tuples = cm.get_cmap(cmap)
    if norm:
        colors = [rgb_tuples(i / n) for i in range(n)]
    else:
        rgb_array = np.asarray([rgb_tuples(i / n) for i in range(n)])
        brg_array = np.zeros(rgb_array.shape)
        brg_array[:,0] = rgb_array[:,2]
        brg_array[:,1] = rgb_array[:,1]
        brg_array[:,2] = rgb_array[:,0]
        colors = [tuple(brg_array[i,:] * 256) for i in range(n)]
    if black:
        black = (0., 0., 0.)
        colors.insert(0, black)
    return colors

def get_graph(hm):
    graph = nx.DiGraph()
    [graph.add_node(i, color=get_node_color(hm.get_space_by_index(i)))
     for i in range(len(hm.spaces) + len(hm.output_spaces))]
    [graph.add_edge(m.origin, m.destination, model=m.model, color=get_edge_color(m))
     for s in hm.spaces[1:] for m in s.incoming if not m.pruned]
    [graph.add_edge(m.origin, m.destination, model=m.model, color=get_edge_color(m))
     for s in hm.output_spaces for m in s.incoming if not m.pruned]
    return graph


def get_node_color(space, colors = ["w", "g", "y"]):
    if space.is_input:
        return colors[0]
    elif space.is_output:
        return colors[1]
    else:
        return colors[2]


def get_edge_color(morphism, colors = ["k", "b", "r"]):
    if morphism.equivariance == "identity":
        return colors[0]
    elif morphism.equivariance == "translations":
        return colors[1]
    else:
        return colors[2]


def visualize_graph(hm, ax = None):
    graph = get_graph(hm)
    if ax is None:
        _, ax = plt.subplots()
    pos = nx.circular_layout(graph)
    n_colors = [n[1] for n in graph.nodes.data('color')]
    nx.draw_networkx_nodes(graph, pos, ax = ax, node_color=n_colors)
    labels = {s.index :s for s in hm.spaces}
    
    for s in hm.output_spaces:
        labels[s.index] = s
    
    nx.draw_networkx_labels(graph, pos, ax = ax, labels = labels)
    e_colors = [e[2] for e in graph.edges.data('color')]
    nx.draw_networkx_edges(graph, pos, arrows=True, ax = ax,
                           edge_color = e_colors)
    lines = {"conv": Line2D([0], [0], color='b', lw=4),
             "identity": Line2D([0], [0], color='k', lw=4)}
    custom_lines = [lines[e] for e in ["conv", "identity"]]
    ax.legend(custom_lines, ["conv", "identity"])
