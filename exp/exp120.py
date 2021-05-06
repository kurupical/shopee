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
import matplotlib.pyplot as plt
import time
from transformers import AdamW, get_linear_schedule_with_warmup
from typing import Tuple, List, Any
import re
import gc

"""
とりあえずこれベースに頑張って写経する
https://www.kaggle.com/tanulsingh077/pytorch-metric-learning-pipeline-only-images
https://www.kaggle.com/zzy990106/b0-bert-cv0-9
"""

DEBUG = False

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


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce

    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


class AdaCos(nn.Module):
    """
    https://github.com/4uiiurz1/pytorch-adacos/blob/master/metrics.py
    """
    def __init__(self, in_features, out_features, m=0.50):
        super(AdaCos, self).__init__()
        self.num_features = in_features
        self.n_classes = out_features
        self.s = math.sqrt(2) * math.log(out_features - 1)
        self.m = m
        self.W = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.W)

    def forward(self, input, label=None):
        # normalize features
        x = F.normalize(input)
        # normalize weights
        W = F.normalize(self.W)
        # dot product
        logits = F.linear(x, W)
        if label is None:
            return logits
        # feature re-scale
        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        with torch.no_grad():
            B_avg = torch.where(one_hot < 1, torch.exp(self.s * logits), torch.zeros_like(logits))
            B_avg = torch.sum(B_avg) / input.size(0)
            # print(B_avg)
            theta_med = torch.median(theta[one_hot == 1])
            self.s = torch.log(B_avg) / torch.cos(torch.min(math.pi/4 * torch.ones_like(theta_med), theta_med))
        output = self.s * logits

        return output


class ArcMarginProductSubcenter(nn.Module):
    def __init__(self, in_features, out_features, k=3):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features * k, in_features))
        self.reset_parameters()
        self.k = k
        self.out_features = out_features

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, features, label=None):
        cosine_all = F.linear(F.normalize(features), F.normalize(self.weight))
        cosine_all = cosine_all.view(-1, self.out_features, self.k)
        cosine, _ = torch.max(cosine_all, dim=2)
        return cosine


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


class Swish(torch.autograd.Function):

    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class SwishModule(nn.Module):
    def forward(self, x):
        return Swish.apply(x)



@dataclasses.dataclass
class Config:
    experiment_name: str = None

    # model
    linear_out: int = 2048
    dropout_nlp: float = 0.5
    dropout_cnn: float = 0.5
    dropout_fc: float = 0.5
    dropout_bert_stack: float = 0.2
    dropout_transformer: float = 0.2
    dropout_cnn_fc: float = 0
    model_name: str = None
    nlp_model_name: str = None
    bert_agg: str = "mean"

    # arcmargin
    m: float = 0.5
    s: float = 32

    # dim
    dim: Tuple[int, int] = (224, 224)

    # optim
    optimizer: Any = AdamW
    optimizer_params = {"weight_decay": 0.1}
    cnn_lr: float = 4e-5
    bert_lr: float = 1e-5
    fc_lr: float = 5e-4
    transformer_lr: float = 1e-3

    scheduler = "get_linear_schedule_with_warmup"
    scheduler_params = {"num_warmup_steps": 1700, "num_training_steps": 1700*10}

    loss: Any = nn.CrossEntropyLoss
    loss_params = {}

    # training
    batch_size: int = 16
    num_workers: int = 2

    if DEBUG:
        epochs: int = 1
    else:
        epochs: int = 10
    early_stop_round: int = 3
    num_classes: int = 11014

    # metric learning
    metric_layer: Any = ArcMarginProduct
    metric_layer_params = {
        "s": s,
        "m": m,
        "in_features": linear_out,
        "out_features": num_classes
    }


    # transformers
    transformer_n_heads: int = 64
    transformer_dropout: float = 0
    transformer_num_layers: int = 1

    # activation
    activation: Any = None

    # debug mode
    debug: bool = DEBUG
    gomi_score_threshold: float = 0.8

    # transforms
    train_transforms: Any = albumentations.Compose([
        albumentations.HorizontalFlip(p=0.5),
        albumentations.ImageCompression(quality_lower=99, quality_upper=100),
        albumentations.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=10, border_mode=0, p=0.7),
        albumentations.Resize(dim[0], dim[1]),
        albumentations.Cutout(max_h_size=int(dim[0] * 0.4), max_w_size=int(dim[0] * 0.4), num_holes=1, p=0.5),
        albumentations.Normalize(),
        ToTensorV2(p=1.0),
    ])
    val_transforms: Any = albumentations.Compose([
            albumentations.Resize(dim[0], dim[1], always_apply=True),
            albumentations.Normalize(),
            ToTensorV2(p=1.0),
    ])

class BertModule(nn.Module):
    def __init__(self,
                 bert,
                 config: Config):
        super().__init__()
        if bert is None:
            self.bert = AutoModel.from_pretrained(config.nlp_model_name)
        else:
            self.bert = bert
        self.config = config
        self.dropout_nlp = nn.Dropout(config.dropout_nlp)
        self.hidden_size = self.bert.config.hidden_size
        self.bert_bn = nn.BatchNorm1d(self.hidden_size)
        self.dropout_stack = nn.Dropout(config.dropout_bert_stack)

    def forward(self, input_ids, attention_mask):
        if "distilbert" in self.config.nlp_model_name:
            text = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0].mean(dim=1)
            text = self.bert_bn(text)
            text = self.dropout_nlp(text)
        else:
            text = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)[2]

            text = torch.stack([self.dropout_stack(x) for x in text[-4:]]).mean(dim=0)
            text = torch.sum(
                text * attention_mask.unsqueeze(-1), dim=1, keepdim=False
            )
            text = text / torch.sum(attention_mask, dim=-1, keepdim=True)
            text = self.bert_bn(text)
            text = self.dropout_nlp(text)

        return text


class ShopeeNet(nn.Module):
    def __init__(self,
                 config: Config,
                 bert=None,
                 pretrained: bool = True):
        super().__init__()
        self.config = config
        self.bert = BertModule(bert=bert,
                               config=config)
        self.bert_fc = nn.Linear(self.bert.hidden_size, config.linear_out)
        self.cnn = create_model(config.model_name,
                                pretrained=pretrained,
                                num_classes=0)
        self.cnn_bn = nn.BatchNorm1d(self.cnn.num_features)
        self.cnn_fc = nn.Linear(self.cnn.num_features, config.linear_out)

        self.fc = nn.Sequential(
            nn.Linear(config.linear_out, config.linear_out),
            nn.BatchNorm1d(config.linear_out)
        )

        self.fc_concat = nn.Sequential(
            nn.Dropout(config.dropout_fc),
            nn.Linear(config.linear_out, config.linear_out),
            nn.BatchNorm1d(config.linear_out)
        )
        self.fc_cnn_out = nn.Sequential(
            nn.Dropout(config.dropout_cnn),
            nn.Linear(config.linear_out, config.linear_out),
            nn.BatchNorm1d(config.linear_out)
        )
        self.fc_text_out = nn.Sequential(
            nn.Dropout(config.dropout_nlp),
            nn.Linear(config.linear_out, config.linear_out),
            nn.BatchNorm1d(config.linear_out)
        )
        self.dropout_cnn = nn.Dropout(config.dropout_cnn)
        self.final = config.metric_layer(**config.metric_layer_params)

    def forward(self, X_image, input_ids, attention_mask, label=None):
        img = self.cnn(X_image)
        img = self.cnn_bn(img)
        img = self.dropout_cnn(img)
        img = self.cnn_fc(img)

        text = self.bert(input_ids, attention_mask)
        text = self.bert_fc(text)
        x = img + text
        ret = self.fc(x)

        # residual
        ret = ret + x

        ret = self.fc_concat(ret)
        img_out = self.fc_cnn_out(img)
        text_out = self.fc_text_out(text)

        ret_img = img_out + img
        ret_text = text_out + text

        if label is not None:
            x = self.final(ret, label)
            img_out = self.final(img_out, label)
            text_out = self.final(text_out, label)
            return x, img_out, text_out, ret, ret_img, ret_text
        else:
            return ret_img, ret_text, ret


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

        return image, input_ids, attention_mask, torch.tensor(row.label_group)


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




def train_fn(dataloader, model, criterion, optimizer, device, scheduler, epoch, ):
    import mlflow

    model.train()
    loss_score = AverageMeter()
    loss_outputs = AverageMeter()
    loss_imgs = AverageMeter()
    loss_texts = AverageMeter()

    tk0 = tqdm.tqdm(enumerate(dataloader), total=len(dataloader))
    for bi, d in tk0:
        batch_size = d[0].shape[0]

        images = d[0].to(device)
        input_ids = d[1].to(device)
        attention_mask = d[2].to(device)
        targets = d[3].to(device)

        optimizer.zero_grad()

        output, img, text, _, _, _ = model(images, input_ids, attention_mask, targets)

        loss_output = criterion(output, targets)
        loss_img = criterion(img, targets)
        loss_text = criterion(text, targets)
        loss = loss_output + loss_img + loss_text
        loss.backward()
        optimizer.step()

        loss_score.update(loss.detach().item(), batch_size)
        loss_outputs.update(loss_output.detach().item(), batch_size)
        loss_imgs.update(loss_img.detach().item(), batch_size)
        loss_texts.update(loss_text.detach().item(), batch_size)
        tk0.set_postfix(LossOutputs=loss_outputs.avg,
                        LossImgs=loss_imgs.avg,
                        LossTexts=loss_texts.avg,
                        Epoch=epoch, LR=optimizer.param_groups[0]['lr'])

        scheduler.step()
        # if not DEBUG:
        #     mlflow.log_metric("train_loss", loss.detach().item())

    return loss_score


def eval_fn(data_loader, model, criterion, device, df_val, epoch, output_dir):
    loss_score = AverageMeter()

    model.eval()
    tk0 = tqdm.tqdm(enumerate(data_loader), total=len(data_loader))

    all_features = []
    all_images = []
    all_texts = []
    all_targets = []
    with torch.no_grad():
        for bi, d in tk0:
            batch_size = d[0].size()[0]

            images = d[0].to(device)
            input_ids = d[1].to(device)
            attention_mask = d[2].to(device)
            targets = d[3].to(device)

            output, _, _, feature, img, text = model(images, input_ids, attention_mask, targets)

            loss = criterion(output, targets)

            loss_score.update(loss.detach().item(), batch_size)
            tk0.set_postfix(Eval_Loss=loss_score.avg)

            all_features.extend(feature.detach().cpu().numpy())
            all_images.extend(img.detach().cpu().numpy())
            all_texts.extend(text.detach().cpu().numpy())
            all_targets.extend(targets.detach().cpu().numpy())

    all_features = np.array(all_features, dtype=np.float32)
    all_images = np.array(all_images, dtype=np.float32)
    all_texts = np.array(all_texts, dtype=np.float32)

    np.save(f"{output_dir}/embeddings_all_epoch{epoch}.npy", all_features)
    np.save(f"{output_dir}/embeddings_images_epoch{epoch}.npy", all_images)
    np.save(f"{output_dir}/embeddings_texts_epoch{epoch}.npy", all_texts)

    del all_images, all_texts
    gc.collect()

    all_features = np.array(all_features, dtype=np.float16)
    best_score, best_th, df_best = get_best_neighbors(df=df_val,
                                                      embeddings=all_features,
                                                      epoch=epoch,
                                                      output_dir=output_dir)

    return loss_score, best_score, best_th, df_best


def get_cv(df):
    tmp = df.groupby('label_group').posting_id.agg('unique').to_dict()
    df['target'] = df.label_group.map(tmp)
    df['f1'] = df.apply(getMetric('pred'), axis=1)
    return df.f1.mean()


def get_best_neighbors(embeddings, df, epoch, output_dir):
    print("threshold search")

    best_th = 0
    best_score = 0

    posting_ids = np.array(df["posting_id"].values.tolist())
    for th in np.arange(0.3, 0.9, 0.025):
        preds = []
        embeddings = torch.tensor(embeddings).cuda()
        embeddings = F.normalize(embeddings)
        cts = torch.matmul(embeddings, embeddings.T)
        for i in range(len(embeddings)):
            IDX = torch.where(cts[i,] > th)[0]
            IDX = IDX.detach().cpu().numpy()
            o = posting_ids[IDX]
            preds.append(o)
        del cts
        gc.collect()
        torch.cuda.empty_cache()
        df["pred"] = preds
        score = get_cv(df)

        if best_score < score:
            best_th = th
            best_score = score
            df_best = df.copy()
        print("th={:.4f} , score={:.4f}".format(th, score))
        del preds
        gc.collect()

    return best_score, best_th, df_best


def main(config, fold=0):
    import mlflow
    mlflow.set_tracking_uri("http://34.121.203.133:5000")  # kiccho_san mlflow

    if config.experiment_name is None:
        raise ValueError
    try:
        seed_torch(19900222)
        output_dir = f"output/{os.path.basename(__file__)[:-3]}/{dt.now().strftime('%Y%m%d%H%M%S')}"
        os.makedirs(output_dir)
        # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=1, eta_min=min_lr, last_epoch=-1)

        df = pd.read_csv("input/shopee-product-matching/train_fold.csv")

        df["title"] = [x.lower() for x in df["title"].values]
        df["filepath"] = df['image'].apply(lambda x: os.path.join('input/shopee-product-matching/', 'train_images', x))
        label_encoder = LabelEncoder()
        df["label_group"] = label_encoder.fit_transform(df["label_group"].values)

        if config.debug:
            df = df.iloc[:100]

        tokenizer = AutoTokenizer.from_pretrained(config.nlp_model_name)

        if DEBUG:
            df = df.iloc[:100]
        if not DEBUG:
            mlflow.start_run(experiment_id=7,
                             run_name=config.experiment_name)
            for key, value in config.__dict__.items():
                mlflow.log_param(key, value)
            mlflow.log_param("fold", fold)

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
                                             {"params": model.bert_fc.parameters(), "lr": config.fc_lr},
                                             {"params": model.cnn.parameters(), "lr": config.cnn_lr},
                                             {"params": model.cnn_bn.parameters(), "lr": config.fc_lr},
                                             {"params": model.cnn_fc.parameters(), "lr": config.fc_lr},
                                             {"params": model.fc_cnn_out.parameters(), "lr": config.fc_lr},
                                             {"params": model.fc_text_out.parameters(), "lr": config.fc_lr},
                                             {"params": model.fc.parameters(), "lr": config.fc_lr},
                                             {"params": model.final.parameters(), "lr": config.fc_lr},
                                             {"params": model.fc_concat.parameters(), "lr": config.fc_lr}],
                                     **config.optimizer_params)
        if config.scheduler == "get_linear_schedule_with_warmup":
            scheduler = get_linear_schedule_with_warmup(optimizer, **config.scheduler_params)
        else:
            scheduler = config.scheduler(optimizer, **config.scheduler_params)
        criterion = config.loss(**config.loss_params)
        best_score = 0
        not_improved_epochs = 0
        for epoch in range(config.epochs):
            train_loss = train_fn(train_loader, model, criterion,
                                  optimizer, device, scheduler=scheduler, epoch=epoch)

            valid_loss, score, best_threshold, df_best = eval_fn(val_loader, model, criterion, device, df_val, epoch, output_dir)

            print(f"CV: {score}")
            if score > best_score:
                print('best model found for epoch {}, {:.4f} -> {:.4f}'.format(epoch, best_score, score))
                best_score = score
                torch.save(model.state_dict(), f'{output_dir}/best_fold{fold}.pth')
                not_improved_epochs = 0
                if not DEBUG:
                    mlflow.log_metric("val_best_cv_score", score, step=epoch)
                df_best.to_csv(f"{output_dir}/df_val_fold{fold}.csv", index=False)
            else:
                not_improved_epochs += 1
                print('{:.4f} is not improved from {:.4f} epoch {} / {}'.format(score, best_score, not_improved_epochs, config.early_stop_round))
                if not_improved_epochs >= config.early_stop_round:
                    print("finish training.")
                    break
            if best_score < config.gomi_score_threshold:
                print("finish training(スコアダメなので打ち切り).")
                break
            if not DEBUG:
                mlflow.log_metric("val_loss", valid_loss.avg, step=epoch)
                mlflow.log_metric("val_cv_score", score, step=epoch)
                mlflow.log_metric("val_best_threshold", best_threshold, step=epoch)

        if not DEBUG:
            mlflow.end_run()
    except Exception as e:
        print(e)
        if not DEBUG:
            mlflow.end_run()


def main_process():

    cfg = Config(experiment_name=f"[add_x_fc]cnn=swin_large_224]/nlp_model=distilbert-indonesian/")
    cfg.model_name = "swin_large_patch4_window7_224"
    cfg.nlp_model_name = "cahya/distilbert-base-indonesian"
    main(cfg)


if __name__ == "__main__":
    main_process()
