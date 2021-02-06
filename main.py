
import numpy as np
from torchvision import datasets
from torchvision.transforms import ToTensor
from update import Global, Local
from dataset import DataClients, ToDataset
from tqdm import tqdm
from options import args_parser


def main():
    args = args_parser()

    data_local_training = datasets.CIFAR10(args.path_cifar10, train=True, download=True)
    data_global_test = datasets.CIFAR10(args.path_cifar10, train=False, transform=ToTensor(), download=True)
    #蒸馏数据集为cifar100
    unlabeled_data = datasets.CIFAR100(args.path_cifar100, transform=ToTensor(), download=True)

    #global进行聚合与teach
    global_model = Global(unlabeled_data=unlabeled_data,
                          num_classes=args.num_classes,
                          num_epochs_global_teaching=args.num_epochs_global_teaching,
                          batch_size_global_teaching=args.batch_size_global_teaching,
                          lr_global_teaching=args.lr_global_teaching,
                          device=args.device)

    #本地进行更新
    local_model = Local(num_classes=args.num_classes,
                        num_epochs_local_training=args.num_epochs_local_training,
                        batch_size_local_training=args.batch_size_local_training,
                        lr_local_training=args.lr_local_training,
                        device=args.device)
    data_clients = DataClients(dataset=data_local_training,
                               num_classes=args.num_classes,
                               num_clients=args.num_clients,
                               ratio_imbalance=args.ratio_imbalance)
    total_clients = list(range(args.num_clients))
    for r in range(args.num_rounds):
        print('EPOCH: [%03d/%d]' % (r + 1, args.num_clients))
        dict_global_params = global_model.download_params()
        online_clients = np.random.choice(total_clients, args.num_online_clients, replace=False)
        list_dicts_local_params = []
        list_nums_local_data = []
        # local training
        for client in tqdm(online_clients, desc='local training'):
            data_client = ToDataset(data_clients[client], ToTensor())
            list_nums_local_data.append(len(data_client))
            local_model.train(dict_global_params, data_client)
            dict_local_params = local_model.upload_params()
            list_dicts_local_params.append(dict_local_params)
        # global update
        global_model.update(list_dicts_local_params, list_nums_local_data)
        # global valuation
        global_model.eval(data_global_test, args.batch_size_test)
        print('-' * 21)


if __name__ == '__main__':
    main()
