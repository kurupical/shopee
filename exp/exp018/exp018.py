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
from sklearn.neighbors import NearestNeighbors
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts, StepLR
from torch.optim import Adam, Optimizer, SGD
import albumentations
from albumentations.pytorch.transforms import ToTensorV2
import dataclasses
import tqdm
from datetime import datetime as dt

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


def getMetric(col):
    def f1score(row):
        n = len( np.intersect1d(row.target,row[col]) )
        return 2*n / (len(row.target)+len(row[col]))
    return f1score


@dataclasses.dataclass
class Config:
    # model
    linear_out = 512
    dropout = 0.5
    model_name = "efficientnet_b3"
    nlp_model_name = "bert-base-multilingual-uncased"
    bert_agg = "mean"

    # arcmargin
    m = 0.5
    s = 32

    # dim
    dim = (512, 512)

    # optim
    optimizer: Optimizer = Adam
    optimizer_params = {}
    base_lr = 5e-4
    bert_lr = 1e-5

    scheduler = ReduceLROnPlateau
    scheduler_params = {"patience": 1, "factor": 0.1, "mode": "max"}

    loss = nn.CrossEntropyLoss
    loss_params = {}

    # training
    batch_size = 16
    num_workers = 1

    epochs = 30
    early_stop_round = 5

    # debug mode
    debug = False

    # transforms
    train_transforms = albumentations.Compose([
            albumentations.Resize(int(dim[0] * 1.1),
                                  int(dim[1] * 1.1), always_apply=True),
            albumentations.CenterCrop(dim[0], dim[1], p=0.5),
            albumentations.Resize(dim[0], dim[1], always_apply=True),
            albumentations.Normalize(),
            ToTensorV2(p=1.0),
    ])
    val_transforms = albumentations.Compose([
            albumentations.Resize(dim[0], dim[1], always_apply=True),
            albumentations.Normalize(),
            ToTensorV2(p=1.0),
    ])

class ShopeeNet(nn.Module):
    def __init__(self,
                 config: Config,
                 bert=None,
                 pretrained: bool = True,
                 num_classes=11014):
        super().__init__()
        self.config = config
        if bert is None:
            self.bert = AutoModel.from_pretrained(config.nlp_model_name)
        else:
            self.bert = bert
        self.bert_bn = nn.BatchNorm1d(self.bert.config.hidden_size)
        self.cnn = create_model(config.model_name,
                                pretrained=pretrained,
                                num_classes=0)
        self.cnn_bn = nn.BatchNorm1d(self.cnn.num_features)

        n_feat_concat = self.cnn.num_features + self.bert.config.hidden_size * 2
        self.fc = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(n_feat_concat, config.linear_out),
            nn.BatchNorm1d(config.linear_out)
        )
        self.final = ArcMarginProduct(s=config.s,
                                      m=config.m,
                                      in_features=config.linear_out,
                                      out_features=num_classes)

    def forward(self, X_image, input_ids, attention_mask, input_ids2, attention_mask2, label=None):
        x = self.cnn(X_image)
        x = self.cnn_bn(x)

        text = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0].mean(axis=1)
        text = self.bert_bn(text)

        text2 = self.bert(input_ids=input_ids2, attention_mask=attention_mask2)[0].mean(axis=1)
        text2 = self.bert_bn(text2)

        x = torch.cat([x, text, text2], dim=1)
        ret = self.fc(x)

        if label is not None:
            x = self.final(ret, label)
            return x, ret
        else:
            return ret

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

        text = self.tokenizer(text, padding="max_length", max_length=128, truncation=True, return_tensors="pt")
        input_ids = text["input_ids"][0]
        attention_mask = text["attention_mask"][0]
        if self.augmentations:
            augmented = self.augmentations(image=image)
            image = augmented['image']

        text2 = row.ocr_text
        text2 = self.tokenizer(text2, padding="max_length", max_length=128, truncation=True, return_tensors="pt")
        input_ids2 = text2["input_ids"][0]
        attention_mask2 = text2["attention_mask"][0]

        return image, input_ids, attention_mask, torch.tensor(row.label_group), input_ids2, attention_mask2


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
    import mlflow

    model.train()
    loss_score = AverageMeter()

    tk0 = tqdm.tqdm(enumerate(dataloader), total=len(dataloader))
    for bi, d in tk0:
        batch_size = d[0].shape[0]

        images = d[0].to(device)
        input_ids = d[1].to(device)
        attention_mask = d[2].to(device)
        targets = d[3].to(device)
        input_ids2 = d[4].to(device)
        attention_mask2 = d[5].to(device)

        optimizer.zero_grad()

        output, _ = model(images, input_ids, attention_mask, input_ids2, attention_mask2, targets)

        loss = criterion(output, targets)

        loss.backward()
        optimizer.step()

        loss_score.update(loss.detach().item(), batch_size)
        tk0.set_postfix(Train_Loss=loss_score.avg, Epoch=epoch, LR=optimizer.param_groups[0]['lr'])

        if scheduler.__class__ != ReduceLROnPlateau:
            scheduler.step()
        mlflow.log_metric("train_loss", loss.detach().item())

    return loss_score


def eval_fn(data_loader, model, criterion, device, df_val):
    loss_score = AverageMeter()

    model.eval()
    tk0 = tqdm.tqdm(enumerate(data_loader), total=len(data_loader))

    all_features = []
    all_targets = []
    with torch.no_grad():
        for bi, d in tk0:
            batch_size = d[0].size()[0]

            images = d[0].to(device)
            input_ids = d[1].to(device)
            attention_mask = d[2].to(device)
            targets = d[3].to(device)
            input_ids2 = d[4].to(device)
            attention_mask2 = d[5].to(device)

            output, feature = model(images, input_ids, attention_mask, input_ids2, attention_mask2, targets)

            loss = criterion(output, targets)

            loss_score.update(loss.detach().item(), batch_size)
            tk0.set_postfix(Eval_Loss=loss_score.avg)

            all_features.extend(feature.detach().cpu().numpy())
            all_targets.extend(targets.detach().cpu().numpy())

    all_features = np.array(all_features, dtype=np.float32)

    best_score, best_th, df_best = get_best_neighbors(df=df_val, embeddings=all_features)

    return loss_score, best_score, best_th, df_best


def get_cv(df):
    tmp = df.groupby('label_group').posting_id.agg('unique').to_dict()
    df['target'] = df.label_group.map(tmp)
    df['f1'] = df.apply(getMetric('pred'),axis=1)
    return df.f1.mean()


def get_best_neighbors(embeddings, df):

    if len(df) > 50:
        model = NearestNeighbors(n_neighbors=50, )
    else:
        model = NearestNeighbors(n_neighbors=len(df) - 1)
    model.fit(embeddings)
    distances, indices = model.kneighbors(embeddings)

    print("threshold search")

    best_th = 0
    best_score = 0

    posting_ids = df["posting_id"].values
    print(f"distances shape: {distances.shape} mean: {distances.mean()}, std: {distances.std()}")
    for th in np.arange(0, 5, 0.1).tolist() + np.arange(5, 10, 0.2).tolist() + np.arange(10, 20, 0.5).tolist() + np.arange(20, 50, 3).tolist():
        preds = []
        for i in range(len(distances)):
            IDX = np.where(distances[i,] < th)[0]
            ids = indices[i, IDX]
            o = posting_ids[ids]
            preds.append(o)
        df["pred"] = preds
        score = get_cv(df)

        if best_score < score:
            best_th = th
            best_score = score
            df_best = df.copy()
        print("th={:.4f} , score={:.4f}".format(th, score))

    return best_score, best_th, df_best


def main(config):
    import mlflow
    seed_torch(19900222)
    output_dir = f"output/{os.path.basename(__file__)[:-3]}/{dt.now().strftime('%Y%m%d%H%M%S')}"
    os.makedirs(output_dir)
    # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=1, eta_min=min_lr, last_epoch=-1)

    df = pd.read_csv("input/shopee-product-matching/train_fold_with_ocr.csv")
    df["filepath"] = df['image'].apply(lambda x: os.path.join('input/shopee-product-matching/', 'train_images', x))
    label_encoder = LabelEncoder()
    df["label_group"] = label_encoder.fit_transform(df["label_group"].values)

    if config.debug:
        df = df.iloc[:100]

    tokenizer = AutoTokenizer.from_pretrained(config.nlp_model_name)

    if config.debug:
        df = df.iloc[:100]
    # for fold in [0, 1, 2, 3, 4]:
    for fold in [2]:
        mlflow.start_run(experiment_id=0,
                         run_name=os.path.basename(__file__))

        df_train = df[df["fold"] != fold]
        df_val = df[df["fold"] == fold]
        # df_train = df[df["label_group"] % 5 != 0]
        # df_val = df[df["label_group"] % 5 == 0]

        train_dataset = ShopeeDataset(df=df_train,
                                      transforms=config.train_transforms,
                                      tokenizer=tokenizer)
        val_dataset = ShopeeDataset(df=df_val,
                                    transforms=config.val_transforms,
                                    tokenizer=tokenizer)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
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
                                             {"params": model.bert_bn.parameters(), "lr": config.bert_lr},
                                             {"params": model.cnn.parameters(), "lr": config.cnn_lr},
                                             {"params": model.cnn_bn.parameters(), "lr": config.cnn_lr},
                                             {"params": model.fc.parameters(), "lr": config.cnn_lr},
                                             {"params": model.final.parameters(), "lr": config.cnn_lr}])
        scheduler = config.scheduler(optimizer, **config.scheduler_params)
        criterion = config.loss(**config.loss_params)

        best_score = 0
        not_improved_epochs = 0
        for epoch in range(config.epochs):
            train_loss = train_fn(train_loader, model, criterion, optimizer, device, scheduler=scheduler, epoch=epoch)

            valid_loss, score, best_threshold, df_best = eval_fn(val_loader, model, criterion, device, df_val)
            scheduler.step(score)

            print(f"CV: {score}")
            if score > best_score:
                print('best model found for epoch {}, {:.4f} -> {:.4f}'.format(epoch, best_score, score))
                best_score = score
                torch.save(model.state_dict(), f'{output_dir}/best_fold{fold}.pth')
                not_improved_epochs = 0
                mlflow.log_metric("val_best_cv_score", score)
                df_best.to_csv(f"{output_dir}/df_val_fold{fold}.csv", index=False)
            else:
                not_improved_epochs += 1
                print('{:.4f} is not improved from {:.4f} epoch {} / {}'.format(score, best_score, not_improved_epochs, config.early_stop_round))
                if not_improved_epochs >= config.early_stop_round:
                    print("finish training.")
                    break
            mlflow.log_param("fold", fold)
            mlflow.log_metric("val_loss", valid_loss.avg)
            mlflow.log_metric("val_cv_score", score)
            mlflow.log_metric("val_best_threshold", best_threshold)

        for key, value in config.__dict__.items():
            mlflow.log_param(key, value)
        mlflow.end_run()

def main_process():
    config = Config()
    main(config)

if __name__ == "__main__":
    main_process()

    """
    for bert_agg in ["max", "min"]:
        config = Config()
        config.bert_agg = bert_agg
        main(config)
    """

