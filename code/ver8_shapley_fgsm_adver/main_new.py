import argparse, json
import datetime
import os
import logging
import torch, random
import matplotlib.pyplot as plt
from LeNet import LeNet5
from server_new import *
from client_new import *
import models, datasets
import numpy as np

def plot_results(results):
    accs, losses = zip(*results)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(accs, label='Accuracy')
    plt.title('Accuracy per Global Epoch')
    plt.xlabel('Global Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(losses, label='Loss')
    plt.title('Loss per Global Epoch')
    plt.xlabel('Global Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()


def select_normal_client(clients):
    normal_clients = [client for client in clients if client.attack_type == 'none']
    selected_client = random.choice(normal_clients)
    return selected_client


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Federated Learning')
    parser.add_argument('-c', '--conf', dest='conf', default='./conf.json')
    args = parser.parse_args()
    with open(args.conf, 'r', encoding='utf-8') as f:
        conf = json.load(f)
    train_datasets, eval_datasets = datasets.get_dataset("../data/", conf["type"])
    server = Server(conf, eval_datasets)
    clients = []

    print(conf["attack_probability"])
    print(conf["adversarial_epsilon"])
    print(conf["adjustment_factor"])
    print(datetime.datetime.now())

    for c in range(conf["no_models"]):
        clients.append(Client(conf, server.global_model, train_datasets, c))

    total_malicious_clients = conf["num_fgsm_attack_clients"] + conf["num_label_poisoning_clients"]
    malicious_indices = random.sample(range(conf["no_models"]), total_malicious_clients)

    fgsm_clients = malicious_indices[:conf["num_fgsm_attack_clients"]]
    label_poisoning_clients = malicious_indices[conf["num_fgsm_attack_clients"]:]

    for i, client in enumerate(clients):
        if i in fgsm_clients:
            client.set_attack_type('fgsm')
        elif i in label_poisoning_clients:
            client.set_attack_type('label_poisoning')
        else:
            client.set_attack_type('none')

    print("对抗性攻击的客户端ID:", fgsm_clients)
    print("标签注毒攻击的客户端ID:", label_poisoning_clients)
    server.set_clients_reputation(clients)

    print("\n")

    global_train_results = []
    normal_client_for_pretrain = select_normal_client(clients)
    pretrain_results = []

    for pretrain_epoch in range(conf["global_epochs"]):
        diff = normal_client_for_pretrain.local_train(server.global_model)
        acc, loss = normal_client_for_pretrain.local_eval(eval_datasets)
        pretrain_results.append((acc, loss))

    print("Pretraining completed.")
    local_accuracies = [acc for acc, _ in pretrain_results]
    local_losses = [loss for _, loss in pretrain_results]
    print("Local Accuracies:")
    for acc in local_accuracies:
        print("{:.2f}".format(acc))
    print("Local Losses:")
    for loss in local_losses:
        print("{:.2f}".format(loss))

    for e in range(conf["global_epochs"]):
        print("Global Epoch %d" % e)
        candidates = random.sample(clients, conf["k"])
        candidates.sort(key=lambda x: x.client_id)
        print("select clients is: ")
        for c in candidates:
            print(c.client_id)
        weight_accumulator = {client.client_id: {} for client in clients}
        for name, params in server.global_model.state_dict().items():
            weight_accumulator[name] = torch.zeros_like(params)
        for c in candidates:
            if c.client_id not in weight_accumulator:
                weight_accumulator[c.client_id] = {}
            diff = c.local_train(server.global_model)
            # 根据客户端返回的参数差值字典更新总体权重
            for name, params in server.global_model.state_dict().items():
                weight_accumulator[c.client_id][name] = diff[name]

        server.model_aggregate(weight_accumulator, candidates)
        acc, loss = server.model_eval()
        global_train_results.append((acc, loss))

        print("Epoch %d, acc: %f, loss: %f\n" % (e, acc, loss))

        original_state_dict = copy.deepcopy(server.global_model.state_dict())

        shapley_values = server.calculate_shapley_values(candidates, server.model_eval)

        server.global_model.load_state_dict(original_state_dict)

        server.update_client_reputation(shapley_values)
        # iter_threshold = 0.80 * (local_accuracies[e] + (1 / local_losses[e]))
        for c in clients:
            client_acc, client_loss = c.local_eval(eval_datasets)
            # server.update_client_reputation(c.client_id, client_acc, client_loss, iter_threshold)
            print("Client %d local eval: acc=%.3f, loss=%.3f, shapley=%.3f, reputation=%.1f"
                  % (c.client_id, client_acc, client_loss,
                     shapley_values[c.client_id],server.client_reputation[c.client_id]))
        print("\n")

    print(datetime.datetime.now())
    plot_results(global_train_results)