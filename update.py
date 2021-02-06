from resnet import resnet14
from light_resnet import light_resnet14
from torch import div, max, eq
from torch.optim import SGD
from torch.nn import CrossEntropyLoss, KLDivLoss
from torch.nn.functional import softmax, log_softmax
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from model import CifarCnn


class Global(object):
    def __init__(self,
                 unlabeled_data,
                 num_classes: int,
                 num_epochs_global_teaching: int,
                 batch_size_global_teaching: int,
                 lr_global_teaching: float,
                 device: str):
        self.model = light_resnet14(num_classes)
        self.model.to(device)
        self.dict_global_params = self.model.state_dict()
        self.unlabeled_data = unlabeled_data
        self.num_epochs_teaching = num_epochs_global_teaching
        self.batch_size_teaching = batch_size_global_teaching
        self.ce_loss = CrossEntropyLoss()
        self.kld_loss = KLDivLoss(reduction='sum')
        self.optimizer = SGD(self.model.parameters(), lr=lr_global_teaching)
        self.device = device

    def update(self, list_dicts_local_params: list, list_nums_local_data: list):
        self.initialize_for_model_fusion(list_dicts_local_params, list_nums_local_data)
        for _ in tqdm(range(self.num_epochs_teaching)):
            self.teach_one_epoch(list_dicts_local_params)

    def teach_one_epoch(self, list_dicts_local_params: list):
        data_loader = DataLoader(self.unlabeled_data, self.batch_size_teaching, shuffle=True)
        for batch_data in data_loader:
            images, labels = batch_data
            images = images.to(self.device)
            logits_teacher = self.avg_logits(images, list_dicts_local_params)
            self.model.load_state_dict(self.dict_global_params)
            self.model.train()
            logits_student = self.model(images)
            x = log_softmax(logits_student, -1)
            y = softmax(logits_teacher.detach(), -1)

            #kl(p||q)＝∑p(logp-logq)
            #kl_loss(x,y)＝∑y(logy-x)
            loss = div(self.kld_loss(x, y), self.batch_size_teaching)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.dict_global_params = self.model.state_dict()

    def initialize_for_model_fusion(self, list_dicts_local_params: list, list_nums_local_data: list):
        for name_param in self.dict_global_params:
            list_values_param = []
            for dict_local_params, num_local_data in zip(list_dicts_local_params, list_nums_local_data):
                list_values_param.append(dict_local_params[name_param] * num_local_data)
            value_global_param = sum(list_values_param) / sum(list_nums_local_data)
            self.dict_global_params[name_param] = value_global_param

    def avg_logits(self, images, list_dicts_local_params: list):
        list_logits = []
        for dict_local_params in list_dicts_local_params:
            self.model.load_state_dict(dict_local_params)
            self.model.eval()
            list_logits.append(self.model(images))

        return sum(list_logits) / len(list_logits)

    def eval(self, data_test, batch_size_test: int):
        self.model.load_state_dict(self.dict_global_params)
        self.model.eval()
        test_loader = DataLoader(data_test, batch_size_test)
        num_corrects = 0
        list_loss = []
        for data_batch in tqdm(test_loader, desc='global testing'):
            images, labels = data_batch
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = self.model(images)
            _, predicts = max(outputs, -1)
            num_corrects += sum(eq(predicts.cpu(), labels.cpu())).item()
            loss_batch = self.ce_loss(outputs, labels)
            list_loss.append(loss_batch.cpu().item())
        accuracy = num_corrects / len(data_test)
        loss = sum(list_loss) / len(list_loss)
        print('acc: %.04f' % accuracy)
        print('loss: %.04f' % loss)

    def download_params(self):
        return self.dict_global_params


class Local(object):
    def __init__(self,
                 num_classes: int,
                 num_epochs_local_training: int,
                 batch_size_local_training: int,
                 lr_local_training: float,
                 device: str):
        self.model = light_resnet14(num_classes)
        self.model.to(device)
        self.num_epochs = num_epochs_local_training
        self.batch_size = batch_size_local_training
        self.ce_loss = CrossEntropyLoss()
        self.optimizer = SGD(self.model.parameters(), lr=lr_local_training)
        self.device = device

    def train(self, dict_global_params, data_client):
        self.model.load_state_dict(dict_global_params)
        self.model.train()
        for epoch in range(self.num_epochs):
            data_loader = DataLoader(dataset=data_client,
                                     batch_size=self.batch_size,
                                     shuffle=True)
            for data_batch in data_loader:
                images, labels = data_batch
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.ce_loss(outputs, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def upload_params(self):
        return self.model.state_dict()
