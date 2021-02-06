from torch.utils.data.dataset import Dataset
from sampling_imbalance import dict_client_idxs_imbalance


class DataClients(object):
    def __init__(self, dataset, num_classes: int, num_clients: int, ratio_imbalance: float):
        self.dataset = dataset
        self.dict_labels_idxs = self.classify_label(num_classes)
        self.dict_client_idxs = dict_client_idxs_imbalance(dict_labels_idxs=self.dict_labels_idxs,
                                                           num_clients=num_clients,
                                                           num_classes=num_classes,
                                                           ratio_imbalance=ratio_imbalance)

    def classify_label(self, num_classes):
        dict_label_idxs = {label: [] for label in range(num_classes)}
        for idx, data in enumerate(self.dataset):
            _, label = data
            dict_label_idxs[label].append(idx)

        return dict_label_idxs

    def __getitem__(self, client):
        idxs = self.dict_client_idxs[client]
        data = []
        for idx in idxs:
            data.append(self.dataset[idx])
        return data

    def __len__(self):
        return len(self.dict_client_idxs)


class ToDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        image, label = self.dataset[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.dataset)


if __name__ == '__main__':
    data_clients = DataClients('/Users/zhouxu/Desktop/FL/FedDF-pytorch/data/CIFAR10', 10, 100, .25)
    for data_client in data_clients:
        pass
