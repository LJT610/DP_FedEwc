import os
import numpy as np
import torch
from torch import nn
import torch.multiprocessing as mp

import config
from core import Agent, Trainer, train_local_mp, callback_func
from model import MNISTModel,CifarModel,GLEAModel, ADNIModel
from data import MNISTData,CifarData,GLEAMData,ADNIData

def ANC(omega,mean_g=0,sigma_0=1.0):
    mean_omega = mean_g^2
    m = mean_omega.size[0]
    beta = np.zeros(m)
    s = np.zeros(m)
    sigma = np.zeros(m)
    new_omega = 0
    for j in range(m):
        #local clip
        beta[j] = mean_omega[j]/np.sum(mean_omega)
        s[j] = beta[j]* mean_omega[j]
        new_omega = np.min(np.abs(omega[j]), np.abs(s[j]))

        # add adaptive noise
        sigma[j] = sigma_0*beta[j]*np.sqrt(m)*mean_omega[j]
        new_omega += np.random.normal(0, sigma[j], 1)

    return new_omega

class MNISTAgent(Agent):
    """MNISTAgent for MNIST and Fashion-MNIST."""
    def __init__(self, global_args, subset=tuple(range(10))):
        super().__init__(global_args, subset, fine='MNIST')
        self.distr_type = global_args.distr_type
        if self.distr_type == 'uniform':
            self.distribution = np.array([0.1] * 10)
        elif self.distr_type == 'dirichlet':
            self.distribution = np.random.dirichlet([global_args.alpha] * 10)
        else:
            raise ValueError(f'Invalid distribution type: {self.distr_type}.')
        self.subset = subset

    def load_data(self, data_alloc, center=False):
        print("=> loading data")
        if center:
            self.train_loader, self.test_loader, self.num_train = data_alloc.create_dataset_for_center(
                self.batch_size, self.num_workers)
        else:
            self.train_loader, self.test_loader, self.num_train = data_alloc.create_dataset_for_client(
                self.distribution, self.batch_size, self.num_workers, self.subset)

    def build_model(self):
        print("=> building model")
        self.model = MNISTModel().to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr,
                                         momentum=0.9, weight_decay=1e-4)
class MNISTTrainer(Trainer):
    """MNIST Trainer."""
    def __init__(self, global_args):
        super().__init__(global_args)
        self.data_alloc = MNISTData(self.num_locals, self.sample_rate)

        # init the global model
        self.global_agent = MNISTAgent(global_args)
        self.global_agent.load_data(self.data_alloc, center=True)
        self.global_agent.build_model()
        self.global_agent.resume_model(self.resume)
        self.a,self.b = None,None
        self.omegas = None

    def build_local_models(self, global_args):
        self.nets_pool = list()
        for _ in range(self.num_locals):
            self.nets_pool.append(MNISTAgent(global_args))
        self.init_local_models()

    def train(self):
        for rnd in range(self.rounds):
            # np.random.shuffle(self.nets_pool)
            num_clients = len(self.nets_pool)
            pool = mp.Pool(num_clients)
            self.q = mp.Manager().Queue()
            self.p = mp.Manager().Queue()
            omegas= list()

            if self.omegas is None:
                self.omegas = [0] * num_clients

            dict_new = self.global_agent.model.state_dict()

            # if self.estimate_weights_in_center and rnd % self.interval == 0:
            #     w_d = self.global_agent.estimate_weights(self.policy)
            # else:
            #     w_d = None

            for i,net in enumerate(self.nets_pool):

                #net.model.load_state_dict(dict_new)
                lr = self.global_agent.lr
                # pool.apply_async(train_local_mp, (net, dict_new,lr,self.local_epochs, rnd, self.q, self.p,self.policy, self.a, self.b,self.omegas[i]))
                omega_process = pool.apply_async(train_local_mp, (net, dict_new,lr,self.local_epochs, rnd, self.q, self.policy, self.a, self.b,self.omegas[i]))
                omegas.append(omega_process)
                # train_local_mp(net, self.local_epochs, rnd, self.q, self.policy, w_d)
            pool.close()
            pool.join()

            self.omegas = list()
            for omega_process in omegas:# update omegas
                omega_result = omega_process.get() #dict
                for k in omega_result:
                    omega_result[k] = omega_result[k].cuda()
                    omega_result[k] = ANC(omega_result[k])
                self.omegas.append(omega_result)

            # while not self.p.empty():
            #     result = self.p.get()
            #     result_cpu = result.cpu()
            #     omegas_results.append(result_cpu)

            self.a,self.b = self.update_global(rnd,self.omegas) #record the a and b at the previous round

class CIFARAgent(Agent):
    """CIFARAgent for CIFAR10 and CIFAR100."""
    def __init__(self, global_args, subset=tuple(range(10)), fine='CIFAR10'):
        super().__init__(global_args, subset, fine)
        self.distr_type = global_args.distr_type
        if self.distr_type == 'uniform':
            self.distribution = np.array([0.1] * 10)
        elif self.distr_type == 'dirichlet':
            self.distribution = np.random.dirichlet([global_args.alpha] * 10)
        else:
            raise ValueError(f'Invalid distribution type: {self.distr_type}.')

    def load_data(self, data_alloc, center=False):
        print("=> loading data")
        if center:
            self.train_loader, self.test_loader, self.num_train = \
                data_alloc.create_dataset_for_center(self.batch_size, self.num_workers)
        else:
            self.train_loader, self.test_loader, self.num_train = \
                data_alloc.create_dataset_for_client(self.distribution, self.batch_size,
                                                     self.num_workers, self.subset)

    def build_model(self):
        print("=> building model")
        if self.fine == 'CIFAR10':
            num_class = 10
        elif self.fine == 'CIFAR100':
            num_class = 100
        else:
            raise ValueError('Invalid dataset choice.')
        self.model = CifarModel(num_class).to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr,
                                         momentum=0.9, weight_decay=1e-4)
class CIFARTrainer(Trainer):
    """CIFAR Trainer."""
    def __init__(self, global_args):
        super().__init__(global_args)
        self.data_alloc = CifarData(self.num_locals, self.sample_rate, fine=self.fine)

        # init the global model
        self.global_agent = CIFARAgent(global_args, fine=self.fine)
        self.global_agent.load_data(self.data_alloc, center=True)
        self.global_agent.build_model()
        self.global_agent.resume_model(self.resume)

    def build_local_models(self, global_args):
        self.nets_pool = list()
        for _ in range(self.num_locals):
            self.nets_pool.append(CIFARAgent(global_args, fine=self.fine))
        self.init_local_models()

    def train(self):
        for rnd in range(self.rounds):
            np.random.shuffle(self.nets_pool)
            pool = mp.Pool(self.num_per_rnd)
            self.q = mp.Manager().Queue()
            dict_new = self.global_agent.model.state_dict()
            if self.estimate_weights_in_center and rnd % self.interval == 0:
                w_d = self.global_agent.estimate_weights(self.policy)
            else:
                w_d = None
            for net in self.nets_pool[:self.num_per_rnd]:
                net.model.load_state_dict(dict_new)
                net.set_lr(self.global_agent.lr)
                pool.apply_async(train_local_mp, (net, self.local_epochs, rnd, self.q, self.policy, w_d))
            pool.close()
            pool.join()
            self.update_global(rnd)

class GLEAMAgent(Agent):
    """GLEAMAgent for Gleam."""
    def __init__(self, global_args, subset=tuple(range(10)), fine='GLEAM'):
        super().__init__(global_args, subset, fine)
        self.distr_type = global_args.distr_type
        if self.distr_type == 'uniform':
            self.distribution = np.array([0.1] * 10)
        elif self.distr_type == 'dirichlet':
            self.distribution = np.random.dirichlet([global_args.alpha] * 10)
        else:
            raise ValueError(f'Invalid distribution type: {self.distr_type}.')

    def load_data(self, data_alloc, center=False):
        print("=> loading data")
        if center:
            self.train_loader, self.test_loader, self.num_train = \
                data_alloc.create_dataset_for_center(self.batch_size, self.num_workers)
        else:
            self.train_loader, self.test_loader, self.num_train = \
                data_alloc.create_dataset_for_client(self.distribution, self.batch_size,
                                                     self.num_workers, self.subset)

    def build_model(self):
        print("=> building model")
        self.model = GLEAMModel(num_class).to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr,
                                         momentum=0.9, weight_decay=1e-4)
class GLEAMTrainer(Trainer):
    """MNIST Trainer."""
    def __init__(self, global_args):
        super().__init__(global_args)
        self.data_alloc = GLEAMData(self.num_locals, self.sample_rate)

        # init the global model
        self.global_agent = GLEAMAgent(global_args)
        self.global_agent.load_data(self.data_alloc, center=True)
        self.global_agent.build_model()
        self.global_agent.resume_model(self.resume)
        self.a,self.b = None,None
        self.omegas = None

    def build_local_models(self, global_args):
        self.nets_pool = list()
        for _ in range(self.num_locals):
            self.nets_pool.append(GLEAMAgent(global_args))
        self.init_local_models()

    def train(self):
        for rnd in range(self.rounds):
            # np.random.shuffle(self.nets_pool)
            num_clients = len(self.nets_pool)
            pool = mp.Pool(num_clients)
            self.q = mp.Manager().Queue()
            self.p = mp.Manager().Queue()
            omegas= list()

            if self.omegas is None:
                self.omegas = [0] * num_clients

            dict_new = self.global_agent.model.state_dict()

            # if self.estimate_weights_in_center and rnd % self.interval == 0:
            #     w_d = self.global_agent.estimate_weights(self.policy)
            # else:
            #     w_d = None

            for i,net in enumerate(self.nets_pool):

                #net.model.load_state_dict(dict_new)
                lr = self.global_agent.lr
                # pool.apply_async(train_local_mp, (net, dict_new,lr,self.local_epochs, rnd, self.q, self.p,self.policy, self.a, self.b,self.omegas[i]))
                omega_process = pool.apply_async(train_local_mp, (net, dict_new,lr,self.local_epochs, rnd, self.q, self.policy, self.a, self.b,self.omegas[i]))
                omegas.append(omega_process)
                # train_local_mp(net, self.local_epochs, rnd, self.q, self.policy, w_d)
            pool.close()
            pool.join()

            self.omegas = list()
            for omega_process in omegas:# update omegas
                omega_result = omega_process.get() #dict
                for k in omega_result:
                    omega_result[k] = omega_result[k].cuda()
                self.omegas.append(omega_result)

            self.a,self.b = self.update_global(rnd,self.omegas) #record the a and b at the previous round

class ADNIAgent(Agent):
    """GLEAMAgent for Gleam."""
    def __init__(self, global_args, subset=tuple(range(10)), fine='ADNI'):
        super().__init__(global_args, subset, fine)
        self.distr_type = global_args.distr_type
        if self.distr_type == 'uniform':
            self.distribution = np.array([0.1] * 10)
        elif self.distr_type == 'dirichlet':
            self.distribution = np.random.dirichlet([global_args.alpha] * 10)
        else:
            raise ValueError(f'Invalid distribution type: {self.distr_type}.')

    def load_data(self, data_alloc, center=False):
        print("=> loading data")
        if center:
            self.train_loader, self.test_loader, self.num_train = \
                data_alloc.create_dataset_for_center(self.batch_size, self.num_workers)
        else:
            self.train_loader, self.test_loader, self.num_train = \
                data_alloc.create_dataset_for_client(self.distribution, self.batch_size,
                                                     self.num_workers, self.subset)

    def build_model(self):
        print("=> building model")
        self.model = ADNIModel(num_class).to(self.device)
        self.criterion = nn.MSELoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr,
                                         momentum=0.9, weight_decay=1e-4)
class ADNITrainer(Trainer):
    """MNIST Trainer."""
    def __init__(self, global_args):
        super().__init__(global_args)
        self.data_alloc = ADNIData(self.num_locals, self.sample_rate)

        # init the global model
        self.global_agent = ADNIAgent(global_args)
        self.global_agent.load_data(self.data_alloc, center=True)
        self.global_agent.build_model()
        self.global_agent.resume_model(self.resume)
        self.a,self.b = None,None
        self.omegas = None

    def build_local_models(self, global_args):
        self.nets_pool = list()
        for _ in range(self.num_locals):
            self.nets_pool.append(ADNIAgent(global_args))
        self.init_local_models()

    def train(self):
        for rnd in range(self.rounds):
            # np.random.shuffle(self.nets_pool)
            num_clients = len(self.nets_pool)
            pool = mp.Pool(num_clients)
            self.q = mp.Manager().Queue()
            self.p = mp.Manager().Queue()
            omegas= list()

            if self.omegas is None:
                self.omegas = [0] * num_clients

            dict_new = self.global_agent.model.state_dict()

            # if self.estimate_weights_in_center and rnd % self.interval == 0:
            #     w_d = self.global_agent.estimate_weights(self.policy)
            # else:
            #     w_d = None

            for i,net in enumerate(self.nets_pool):

                #net.model.load_state_dict(dict_new)
                lr = self.global_agent.lr
                # pool.apply_async(train_local_mp, (net, dict_new,lr,self.local_epochs, rnd, self.q, self.p,self.policy, self.a, self.b,self.omegas[i]))
                omega_process = pool.apply_async(train_local_mp, (net, dict_new,lr,self.local_epochs, rnd, self.q, self.policy, self.a, self.b,self.omegas[i]))
                omegas.append(omega_process)
                # train_local_mp(net, self.local_epochs, rnd, self.q, self.policy, w_d)
            pool.close()
            pool.join()

            self.omegas = list()
            for omega_process in omegas:# update omegas
                omega_result = omega_process.get() #dict
                for k in omega_result:
                    omega_result[k] = omega_result[k].cuda()
                self.omegas.append(omega_result)

            # while not self.p.empty():
            #     result = self.p.get()
            #     result_cpu = result.cpu()
            #     omegas_results.append(result_cpu)

            self.a,self.b = self.update_global(rnd,self.omegas) #record the a and b at the previous round


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    mp.set_start_method('forkserver')
    if args.fine == 'MNIST':
        trainer = MNISTTrainer(args)
    elif args.fine == 'CIFAR10':
        trainer = CIFARTrainer(args)
    elif args.fine == 'ADNI':
        trainer = ADNITrainer(args)
    elif args.fine == 'GLEAM':
        trainer = GLEAMTrainer(args)
    else:
        raise ValueError("No available dataset")
    # test
    if args.mode == 'test':
        trainer.test()
        return

    trainer.build_local_models(args)
    trainer.train()

if __name__ == '__main__':
    args = config.get_args()
    main(args)