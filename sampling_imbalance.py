import numpy as np


def partition_balance(idxs, num_split: int):

    num_per_part, r = len(idxs) // num_split, len(idxs) % num_split
    parts = []
    i, r_used = 0, 0
    while i < len(idxs):
        if r_used < r:
            parts.append(idxs[i:(i + num_per_part + 1)])
            i += num_per_part + 1
            r_used += 1
        else:
            parts.append(idxs[i:(i + num_per_part)])
            i += num_per_part

    return parts


def partition_imbalance(idxs, ratio_imbalance: float):

    num_min = round(len(idxs) * ratio_imbalance / (ratio_imbalance + 1))

    idxs_min = idxs[:num_min]
    idxs_maj = idxs[num_min:]

    return idxs_min, idxs_maj


def split(dict_labels_idxs: dict, num_clients: int, num_classes: int, ratio_imbalance: float, random: bool):
    dict_labels_idxs_split = {label: {'idxs_min': None, 'idxs_maj': None} for label in range(num_classes)}
    for label in dict_labels_idxs:
        if random:
            np.random.shuffle(dict_labels_idxs[label])
        idxs_min, idxs_maj = partition_imbalance(dict_labels_idxs[label], ratio_imbalance=ratio_imbalance)
        dict_labels_idxs_split[label]['idxs_min'] = partition_balance(idxs_min, num_clients // num_classes)
        dict_labels_idxs_split[label]['idxs_maj'] = partition_balance(idxs_maj, num_clients // num_classes)

    return dict_labels_idxs_split


def clients_choose_labels(num_clients: int, num_classes: int):
    dict_clients_labels = {client: {'label_min': None, 'label_maj': None} for client in range(num_clients)}
    client = 0
    for i in range(num_clients // num_classes):
        labels_min = list(range(num_classes))
        labels_maj = list(range(num_classes))
        shuffle = True
        while shuffle:
            np.random.shuffle(labels_min)
            np.random.shuffle(labels_maj)
            if sum(np.array(labels_min) == np.array(labels_maj)) == 0:
                shuffle = False
        for label_min, label_maj in zip(labels_min, labels_maj):
            dict_clients_labels[client]['label_min'] = label_min
            dict_clients_labels[client]['label_maj'] = label_maj
            client += 1

    return dict_clients_labels


def clients_choose_idxs(dict_labels_idxs_split: dict, dict_clients_labels: dict, num_clients: int):
    dict_clients_idxs = {client: {'idxs_min': None, 'idxs_maj': None} for client in range(num_clients)}
    for client in dict_clients_labels:
        label_min = dict_clients_labels[client]['label_min']
        label_maj = dict_clients_labels[client]['label_maj']
        dict_clients_idxs[client]['idxs_min'] = dict_labels_idxs_split[label_min]['idxs_min'].pop()
        dict_clients_idxs[client]['idxs_maj'] = dict_labels_idxs_split[label_maj]['idxs_maj'].pop()

    return dict_clients_idxs


def concat_min_maj(dict_clients_idxs: dict):
    dict_idxs = {}
    for client in dict_clients_idxs:
        dict_idxs[client] = dict_clients_idxs[client]['idxs_min'] + dict_clients_idxs[client]['idxs_maj']

    return dict_idxs


# {first label: [indexes of first label],..., last label: [indexes of last label]} =>
# {first client: [indexes of first client],..., last client: [indexes of last client]}
def dict_client_idxs_imbalance(dict_labels_idxs: dict, num_clients: int, num_classes: int, ratio_imbalance: float):
    dict_labels_idxs_split = split(dict_labels_idxs, num_clients, num_classes, ratio_imbalance, True)
    dict_clients_labels = clients_choose_labels(num_clients, num_classes)
    dict_clients_idxs = clients_choose_idxs(dict_labels_idxs_split, dict_clients_labels, num_clients)
    return concat_min_maj(dict_clients_idxs)
