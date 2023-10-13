# -*- coding: utf-8 -*-
import os
import datetime
import copy
import collections
from queue import Queue
from abc import ABC, abstractmethod
import numpy as np
import torch
from torch import nn
from tensorboardX import SummaryWriter
import pdb

def train_local_mp(net,dict_new, lr, local_epochs, rnd, q, policy, a,b,omega):
    """For multiprocessing train"""
    print(f"=> Train begins: Round {rnd}")
    # if policy in ['ewc', 'mas'] and w_d is None:
    #     w_d = net.estimate_weights(policy)

    # net.global_model = copy.deepcopy(net.model.state_dict())

    # compute the last a and b for each client
    if (a is not None) and (b is not None):
        a_c, b_c = dict(), dict()
        for k, v in net.model.state_dict().items():
            a_c[k] = a[k] - omega[k]
            b_c[k] = b[k] - omega[k] * v
    else:
        a_c, b_c = None, None

    net.model.load_state_dict(dict_new)
    net.set_lr(lr)
    for _ in range(local_epochs):
        net.train(a_c,b_c)
        # pdb.set_trace()  # 在train_local_mp函数中设置断点

    if policy in ['ewc', 'mas']:
        omega = net.estimate_weights(policy) #a for local client

    print(f"=> Test begins: Round {rnd}")
    test_acc, test_loss = net.test()
    q.put((test_acc, test_loss))
    # p.put(omega)

    return omega

def callback_func(omega):
    return omega

def test_local_mp(net, q, i):
    """For multiprocessing test"""
    print(f"=> Clients {i} Test Begins.")
    test_acc, test_loss = net.test()
    q.put((test_acc, test_loss))

class Trainer(ABC):
    """Base Trainer Class"""
    def __init__(self, global_args):
        self.fine = global_args.fine
        self.num_locals = global_args.num_locals
        self.num_per_rnd = global_args.num_per_rnd
        self.local_epochs = global_args.local_epochs
        self.rounds = global_args.rounds
        self.policy = global_args.policy
        self.sample_rate = global_args.sample_rate
        self.interval = global_args.interval
        self.resume = global_args.resume
        self.log_dir = global_args.log_dir
        self.estimate_weights_in_center = global_args.estimate_weights_in_center

        self.device = torch.device('cuda:{}'.format(global_args.gpu))
        self.data_alloc = None
        self.writer = None
        self.global_agent = None
        self.nets_pool = None
        self.q = Queue()
        self.local_attn_weights = list()

        self.writer = None
        if global_args.mode == 'train':
            writer_dir = os.path.join(f'{self.log_dir}/{self.fine}_{self.policy}_{self.num_locals}',
                                      datetime.datetime.now().strftime('%b%d_%H-%M'))
            self.writer = SummaryWriter(writer_dir)

    def __del__(self):
        if self.writer is not None:
            self.writer.close()

    def init_local_models(self):
        # duplicate the global model to local nets
        global_state = self.global_agent.model.state_dict()
        for net in self.nets_pool:
            net.load_data(self.data_alloc)
            net.build_model()
            net.model.load_state_dict(global_state)
        print(f'=> {len(self.nets_pool)} local nets init done.')

    def model_aggregation_avg(self,omegas):
        # compute average of models
        print('=> model aggregation with policy (avg)')
        dict_new = collections.defaultdict(list)
        # num of train examples of each agent
        weights = list()

        #compute a and b terms
        a , b= collections.defaultdict(list),collections.defaultdict(list)

        for i,omega in enumerate(omegas):
            for k,v in omega.items():
                a[k].append(v)

        for i,net in enumerate(self.nets_pool):
            weights.append(net.num_train)
            for k, v in net.model.state_dict().items():
                dict_new[k].append(v)
                b[k].append(a[k][i] * v)

        # normalization
        weights = torch.as_tensor(weights, dtype=torch.float, device=self.device)
        weights.div_(weights.sum())
        for k, v in dict_new.items():
            v = torch.stack(v)
            expected_shape = [self.num_per_rnd] + [1] * (v.dim() - 1)
            dict_new[k] = torch.sum(v.mul_(weights.reshape(expected_shape)), dim=0)

        for k, v in a.items():
            v = torch.stack(v)
            # expected_shape = [self.num_per_rnd] + [1] * (v.dim() - 1)
            a[k] = torch.sum(v, dim=0)

        for k, v in b.items():
            v = torch.stack(v)
            # expected_shape = [self.num_per_rnd] + [1] * (v.dim() - 1)
            b[k] = torch.sum(v, dim=0)

        return dict_new,a,b

    def update_global(self, rnd, omegas):
        dict_new,a,b = self.model_aggregation_avg(omegas)

        # update global model and test
        self.global_agent.model.load_state_dict(dict_new)
        self.global_agent.update_lr(rnd, self.writer)
        print(f"=> Global Test begins: Round {rnd}")
        global_acc, global_loss = self.global_agent.test(rnd)
        self.writer.add_scalar('global/accuracy', global_acc, rnd)
        self.writer.add_scalar('global/loss', global_loss, rnd)

        local_test = list()
        while not self.q.empty():
            local_test.append(self.q.get())
        local_acc, local_loss = np.mean(np.asarray(local_test), axis=0)
        local_acc_std, local_loss_std = np.std(np.asarray(local_test), axis=0)
        self.writer.add_scalar('local/accuracy', local_acc, rnd)
        self.writer.add_scalar('local/loss', local_loss, rnd)
        self.writer.add_scalar('local/accuracy_std', local_acc_std, rnd)
        self.writer.add_scalar('local/loss_std', local_loss_std, rnd)
        #
        self.global_agent.maybe_save(rnd, local_acc)
        return a,b

    def test(self):
        print("=> Test begins.")
        self.global_agent.test()

    @abstractmethod
    def build_local_models(self, global_args):
        pass

    @abstractmethod
    def train(self):
        pass
