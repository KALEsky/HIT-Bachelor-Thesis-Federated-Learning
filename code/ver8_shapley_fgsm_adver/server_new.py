import models
import torch, math
import itertools
import numpy as np


class Server(object):
    def __init__(self, conf, eval_dataset, clients=None):
        self.conf = conf
        self.global_model = models.get_model(self.conf["model_name"])
        self.eval_loader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=self.conf["batch_size"],
            shuffle=True
        )
        self.clients = clients if clients is not None else []
        self.client_reputation = {}

    def set_clients_reputation(self, clients):
        self.clients = clients
        self.client_reputation = {client.client_id: 6 for client in self.clients}

    def update_client_reputation(self, shapley_values):
        # 计算Shapley值的33分位数和66分位数
        shapley_values_list = list(shapley_values.values())
        q33, q66 = np.percentile(shapley_values_list, [33, 66])

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        for client_id, shapley_value in shapley_values.items():
            if shapley_value > q66:
                deviation = (shapley_value - q66) / (max(shapley_values_list) - q66)
                adjustment = sigmoid(deviation) - 0.5
            elif shapley_value < q33:
                deviation = (q33 - shapley_value) / (q33 - min(shapley_values_list))
                adjustment = -(sigmoid(deviation) - 0.5)
            else:
                adjustment = 0

            self.client_reputation[client_id] += adjustment * self.conf["adjustment_factor"]

            self.client_reputation[client_id] = max(self.conf["min_reputation"],
                                                    min(self.conf["max_reputation"], self.client_reputation[client_id]))

    # def update_client_reputation(self, client_id, client_acc, client_loss, iter_threshold):
    #     combined_metric = client_acc + (1 / client_loss)
    #     if combined_metric >= iter_threshold:
    #         self.client_reputation[client_id] += self.conf["reward_factor"]
    #     else:
    #         self.client_reputation[client_id] -= self.conf["penalty_factor"]
    #     self.client_reputation[client_id] = max(self.conf["min_reputation"],
    #                                             min(self.conf["max_reputation"], self.client_reputation[client_id]))

    def model_aggregate(self, weight_accumulator, clients):
        for name, data in self.global_model.state_dict().items():
            total_reputation = sum(self.client_reputation[client.client_id] for client in clients)
            aggregated_update = sum(
                client_weight * weight_accumulator[client.client_id][name]
                for client, client_weight in
                zip(clients, (self.client_reputation[client.client_id] / total_reputation for client in clients))
            )
            update_per_layer = aggregated_update * self.conf["lambda"]
            if data.type() != update_per_layer.type():
                data.add_(update_per_layer.to(torch.int64))
            else:
                data.add_(update_per_layer)

    def model_eval(self):
        self.global_model.eval()
        total_loss = 0.0
        correct = 0
        dataset_size = 0
        for batch_id, batch in enumerate(self.eval_loader):
            data, target = batch
            dataset_size += data.size()[0]
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
            output = self.global_model(data)
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

    def calculate_shapley_values(self, clients, model_eval_func):
        shapley_values = {client.client_id: 0 for client in clients}
        total_clients = len(clients)
        factorial_total_clients = math.factorial(total_clients)

        memo = {}

        for client in clients:
            client_id = client.client_id
            # 遍历除当前客户端外的所有客户端组合
            for r in range(total_clients):
                subsets = itertools.combinations([c for c in clients if c.client_id != client_id], r)
                for subset in subsets:
                    subset_key = tuple(sorted([c.client_id for c in subset]))
                    with_client = list(subset) + [client]
                    without_client = list(subset)

                    if subset_key in memo:
                        without_client_perf = memo[subset_key]
                    else:
                        acc, loss = self.model_eval_with_clients(without_client, model_eval_func)
                        without_client_perf = acc + 1 / loss
                        memo[subset_key] = without_client_perf

                    with_subset_key = tuple(sorted(list(subset_key) + [client_id]))
                    if with_subset_key in memo:
                        with_client_perf = memo[with_subset_key]
                    else:
                        acc, loss = self.model_eval_with_clients(with_client, model_eval_func)
                        with_client_perf = acc + 1 / loss
                        memo[with_subset_key] = with_client_perf

                    shapley_values[client_id] += (with_client_perf - without_client_perf) / (
                            math.comb(total_clients - 1, r) * factorial_total_clients)

        return shapley_values

    def model_eval_with_clients(self, clients, model_eval_func):
        # 重置全局模型为初始状态
        self.global_model = models.get_model(self.conf["model_name"])

        # 初始化weight_accumulator
        weight_accumulator = {name: torch.zeros_like(param) for name, param in self.global_model.state_dict().items()}

        # 计算每个客户端的贡献并累加到weight_accumulator中
        for client in clients:
            diff = client.local_train(self.global_model)
            for name, value in diff.items():
                weight_accumulator[name] += value

        # 将累加的更新应用到全局模型上
        for name, data in self.global_model.state_dict().items():
            data.add_(weight_accumulator[name], alpha=self.conf["lambda"])

        # 使用提供的model_eval_func函数评估模型性能
        acc, loss = model_eval_func()
        return acc, loss


'''

    def calculate_shapley_values(self, clients, model_eval_func):
        shapley_values = {client.client_id: 0 for client in clients}
        total_clients = len(clients)
        factorial_total_clients = math.factorial(total_clients)

        for client in clients:
            client_id = client.client_id
            # 遍历除当前客户端外的所有客户端组合
            for r in range(total_clients):
                subsets = itertools.combinations([c for c in clients if c.client_id != client_id], r)
                for subset in subsets:
                    with_client = list(subset) + [client]
                    without_client = list(subset)

                    # 计算包含当前客户端和不包含当前客户端的模型性能
                    with_client_perf = self.model_eval_with_clients(with_client, model_eval_func)
                    without_client_perf = self.model_eval_with_clients(without_client, model_eval_func)

                    # 更新Shapley值
                    shapley_values[client_id] += (with_client_perf - without_client_perf) / (
                            math.comb(total_clients - 1, r) * factorial_total_clients)

        return shapley_values

    def model_eval_with_clients(self, clients, model_eval_func):
        # 重置全局模型为初始状态
        self.global_model = models.get_model(self.conf["model_name"])

        # 初始化weight_accumulator
        weight_accumulator = {name: torch.zeros_like(param) for name, param in self.global_model.state_dict().items()}

        # 计算每个客户端的贡献并累加到weight_accumulator中
        for client in clients:
            diff = client.local_train(self.global_model)
            for name, value in diff.items():
                weight_accumulator[name] += value

        # 将累加的更新应用到全局模型上
        for name, data in self.global_model.state_dict().items():
            data.add_(weight_accumulator[name], alpha=self.conf["lambda"])

        # 使用提供的model_eval_func函数评估模型性能
        acc, loss = model_eval_func()
        return acc

'''
