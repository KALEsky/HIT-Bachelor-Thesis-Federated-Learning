import copy
import models
import torch
from torchattacks import FGSM


class Client(object):
    def __init__(self, conf, model, train_dataset, id=-1):
        self.conf = conf
        if self.conf["model_name"] == "resnet18":
            self.local_model = models.get_model(self.conf["model_name"])
        else:
            self.local_model = models.get_model(self.conf["model_name"])
        self.client_id = id
        self.train_dataset = train_dataset
        all_range = list(range(len(self.train_dataset)))
        data_len = int(len(self.train_dataset) / self.conf['no_models'])
        train_indices = all_range[id * data_len: (id + 1) * data_len]
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=conf["batch_size"],
            sampler=torch.utils.data.sampler.SubsetRandomSampler(train_indices)
        )
        self.attack_type = 'none'

    def set_attack_type(self, attack_type):
        self.attack_type = attack_type

    def local_train(self, model):
        for name, param in model.state_dict().items():
            self.local_model.state_dict()[name].copy_(param.clone())
        optimizer = torch.optim.SGD(
            self.local_model.parameters(),
            lr=self.conf['lr'],
            momentum=self.conf['momentum']
        )
        self.local_model.train()
        if self.attack_type == 'fgsm':
            fgsm = FGSM(self.local_model, eps=self.conf["adversarial_epsilon"])
            for e in range(self.conf["local_epochs"]):
                for batch_id, batch in enumerate(self.train_loader):
                    data, target = batch
                    if torch.cuda.is_available():
                        data = data.cuda()
                        target = target.cuda()
                    adversarial_samples = fgsm(data, target)
                    optimizer.zero_grad()
                    output = self.local_model(adversarial_samples)
                    loss = torch.nn.functional.cross_entropy(output, target)
                    loss.backward()
                    optimizer.step()
        elif self.attack_type == 'label_poisoning':
            for e in range(self.conf["local_epochs"]):
                for batch_id, batch in enumerate(self.train_loader):
                    data, target = batch
                    if torch.cuda.is_available():
                        data = data.cuda()
                        target = target.cuda()
                    optimizer.zero_grad()
                    attack_mask = torch.rand(target.size()) < self.conf["attack_probability"]
                    wrong_labels = torch.randint(0, self.conf["num_classes"], target.size())
                    target[attack_mask] = wrong_labels[attack_mask]
                    output = self.local_model(data)
                    loss = torch.nn.functional.cross_entropy(output, target)
                    loss.backward()
                    optimizer.step()
        else:
            for e in range(self.conf["local_epochs"]):
                for batch_id, batch in enumerate(self.train_loader):
                    data, target = batch
                    if torch.cuda.is_available():
                        data = data.cuda()
                        target = target.cuda()
                    optimizer.zero_grad()
                    output = self.local_model(data)
                    loss = torch.nn.functional.cross_entropy(output, target)
                    loss.backward()
                    optimizer.step()
                # print("Epoch %d done." % e)
        diff = dict()
        for name, data in self.local_model.state_dict().items():
            diff[name] = (data - model.state_dict()[name])
        return diff

    def local_eval(self, eval_dataset):
        eval_loader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=self.conf["batch_size"],
            shuffle=True
        )
        total_loss = 0.0
        correct = 0
        dataset_size = 0
        self.local_model.eval()
        for batch_id, batch in enumerate(eval_loader):
            data, target = batch
            dataset_size += data.size()[0]
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
            output = self.local_model(data)
            total_loss += torch.nn.functional.cross_entropy(
                output,
                target,
                reduction='sum'
            ).item()
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
        acc = 100.0 * (float(correct) / float(dataset_size))
        total_l = total_loss / dataset_size
        return acc, total_l
