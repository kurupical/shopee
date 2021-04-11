import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
from timm import create_model
import math
import cv2
import random
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts, StepLR
from torch.optim import Adam, Optimizer, SGD
import albumentations
from albumentations.pytorch.transforms import ToTensorV2
import dataclasses
import tqdm
from datetime import datetime as dt
import mlflow

"""
とりあえずこれベースに頑張って写経する
https://www.kaggle.com/tanulsingh077/pytorch-metric-learning-pipeline-only-images
https://www.kaggle.com/zzy990106/b0-bert-cv0-9
"""

def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

@dataclasses.dataclass
class Config:
    # model
    linear_out = 1024
    dropout = 0.5
    model_name = "efficientnet_b2"

    # arcmargin
    m = 0.5
    s = 30

    # dim
    dim = (256, 256)

    # optim
    optimizer: Optimizer = Adam
    optimizer_params = {}
    bert_lr = 1e-5
    base_lr = 1e-3

    scheduler = StepLR
    scheduler_params = {"step_size": 1, "gamma": 0.9995}

    loss = nn.CrossEntropyLoss
    loss_params = {}

    # training
    batch_size = 16
    num_workers = 1

    epochs = 1
    early_stop_round = 5

    # debug mode
    debug = True


class ShopeeNet(nn.Module):
    def __init__(self,
                 config: Config,
                 num_classes=11014):
        super().__init__()
        self.bert = AutoModel.from_pretrained("bert-base-uncased")
        self.activation = nn.ReLU()
        self.cnn = create_model(config.model_name,
                                pretrained=True,
                                num_classes=0)
        self.fc = nn.Linear(self.cnn.num_features + self.bert.config.hidden_size, config.linear_out)
        self.final = ArcMarginProduct(s=config.s,
                                      m=config.m,
                                      in_features=config.linear_out,
                                      out_features=num_classes)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, X_image, input_ids, attention_mask, label=None):
        x = self.cnn(X_image)
        text = self.bert(input_ids=input_ids, attention_mask=attention_mask)[1]

        x = torch.cat([x, text], dim=1)
        x = self.fc(x)
        x = self.dropout(x)
        x = self.activation(x)

        x = self.final(x, label)
        return x


class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features, s, m, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)

        return output


class ShopeeDataset(Dataset):
    def __init__(self, df, tokenizer, transforms=None):
        self.df = df.reset_index()
        self.augmentations = transforms
        self.tokenizer = tokenizer

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index]

        text = row.title

        image = cv2.imread(row.filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        text = self.tokenizer(text, padding="max_length", truncation=True, return_tensors="pt")
        input_ids = text["input_ids"][0]
        attention_mask = text["attention_mask"][0]
        if self.augmentations:
            augmented = self.augmentations(image=image)
            image = augmented['image']

        return image, input_ids, attention_mask, torch.tensor(row.label_group)



def get_train_transforms(config: Config):
    return albumentations.Compose(
        [
            albumentations.Resize(config.dim[0], config.dim[1], always_apply=True),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.Rotate(limit=120, p=0.8),
            albumentations.Normalize(),
            ToTensorV2(p=1.0),
        ]
    )

def get_valid_transforms(config: Config):

    return albumentations.Compose(
        [
            albumentations.Resize(config.dim[0], config.dim[1], always_apply=True),
            albumentations.Normalize(),
            ToTensorV2(p=1.0)
        ]
    )

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_fn(dataloader, model, criterion, optimizer, device, scheduler, epoch):
    model.train()
    loss_score = AverageMeter()

    tk0 = tqdm.tqdm(enumerate(dataloader), total=len(dataloader))
    for bi, d in tk0:
        batch_size = d[0].shape[0]

        images = d[0].to(device)
        input_ids = d[1].to(device)
        attention_mask = d[2].to(device)
        targets = d[3].to(device)

        optimizer.zero_grad()

        output = model(images, input_ids, attention_mask, targets)

        loss = criterion(output, targets)

        loss.backward()
        optimizer.step()

        loss_score.update(loss.detach().item(), batch_size)
        tk0.set_postfix(Train_Loss=loss_score.avg, Epoch=epoch, LR=optimizer.param_groups[0]['lr'])

        scheduler.step()
        mlflow.log_metric("train_loss", loss.detach().item())

    return loss_score


def eval_fn(data_loader, model, criterion, device):
    loss_score = AverageMeter()

    model.eval()
    tk0 = tqdm.tqdm(enumerate(data_loader), total=len(data_loader))

    with torch.no_grad():
        for bi, d in tk0:
            batch_size = d[0].size()[0]

            images = d[0].to(device)
            input_ids = d[1].to(device)
            attention_mask = d[2].to(device)
            targets = d[3].to(device)

            output = model(images, input_ids, attention_mask, targets)

            loss = criterion(output, targets)

            loss_score.update(loss.detach().item(), batch_size)
            tk0.set_postfix(Eval_Loss=loss_score.avg)

    return loss_score


def main(config):
    seed_torch(19900222)
    output_dir = f"output/{os.path.basename(__file__)[:-3]}/{dt.now().strftime('%Y%m%d%H%M%S')}"
    os.makedirs(output_dir)
    # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=1, eta_min=min_lr, last_epoch=-1)

    df = pd.read_csv("input/shopee-product-matching/train.csv")
    df["filepath"] = df['image'].apply(lambda x: os.path.join('input/shopee-product-matching/', 'train_images', x))
    label_encoder = LabelEncoder()
    df["label_group"] = label_encoder.fit_transform(df["label_group"].values)

    if config.debug:
        df = df.iloc[:100]
    kfold = KFold(5, shuffle=True, random_state=19900222)


    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    for i, (train_idx, val_idx) in enumerate(kfold.split(df)):
        mlflow.start_run(experiment_id=0,
                         run_name=os.path.basename(__file__))

        df_train = df.iloc[train_idx].reset_index(drop=True)
        df_val = df.iloc[val_idx].reset_index(drop=True)

        train_dataset = ShopeeDataset(df=df_train,
                                      transforms=get_train_transforms(config=config),
                                      tokenizer=tokenizer)
        val_dataset = ShopeeDataset(df=df_val,
                                    transforms=get_valid_transforms(config=config),
                                    tokenizer=tokenizer)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            pin_memory=True,
            drop_last=True,
            num_workers=config.num_workers
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )

        device = torch.device("cuda")

        model = ShopeeNet(config=config)
        model.to("cuda")
        optimizer = config.optimizer(params=[{"params": model.bert.parameters(), "lr": config.bert_lr},
                                             {"params": model.cnn.parameters(), "lr": config.base_lr},
                                             {"params": model.fc.parameters(), "lr": config.base_lr},
                                             {"params": model.final.parameters(), "lr": config.base_lr}])
        scheduler = config.scheduler(optimizer, **config.scheduler_params)
        criterion = config.loss(**config.loss_params)

        best_loss = 10000
        not_improved_epochs = 0
        for epoch in range(config.epochs):
            train_loss = train_fn(train_loader, model, criterion, optimizer, device, scheduler=scheduler, epoch=epoch)

            valid_loss = eval_fn(val_loader, model, criterion, device)
            scheduler.step(valid_loss.avg)

            if valid_loss.avg < best_loss:
                best_loss = valid_loss.avg
                torch.save(model.state_dict(), f'{output_dir}/best.pth')
                not_improved_epochs = 0
                print('best model found for epoch {}, {:.4f} -> {:.4f}'.format(epoch, best_loss, valid_loss.avg))
            else:
                not_improved_epochs += 1
                print('not improved from {:.4f} epoch {} / {}'.format(valid_loss.avg, not_improved_epochs, config.early_stop_round))
                if not_improved_epochs >= config.early_stop_round:
                    print("finish training.")
                    break
            mlflow.log_metric("val_loss", valid_loss.avg)

        for key, value in config.__dict__.items():
            mlflow.log_param(key, value)
        mlflow.log_metric("param", i)
        mlflow.end_run()

        break


if __name__ == "__main__":
    """
    for lr in [1e-4, 5e-4, 1e-2]:
        for s in [8, 16, 32]:
            config = Config()
            config.base_lr = lr
            config.bert_lr = lr * 0.01
            config.s = s
            config.debug = False
            main(config)
    """
    """
    for bert_lr in [1e-4, 1e-5, 1e-6]:
        config = Config()
        config.base_lr = 1e-3
        config.bert_lr = bert_lr
        config.debug = False
        main(config)

    config = Config()
    for m in [0.1, 0.5, 1]:
        config.m = m
        config.debug = False
        main(config)
    """

    for dropout in [0, 0.1, 0.2, 0.3]:
        config = Config()
        config.dropout = dropout
        config.debug = False
        main(config)

    for model_name in ["resnet18", "resnet34", "resnet50",
                       "efficientnet_b0", "efficientnet_b1", "efficientnet_b2",
                       "efficientnet_b3", "efficientnet_b4", "efficientnet_b5",
                       "efficientnet_b6", "efficientnet_b7"]:
        try:
            config = Config()
            config.model_name = model_name
            config.debug = False
            main(config)
        except Exception as e:
            print(e)