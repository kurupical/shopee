import sys
sys.path.append("/kaggle/input/shopee-exp/exp")
import exp097_3out
import exp103_3out
import exp095_3out
import exp102_3out
import exp104_3out
import torch
import tqdm
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoConfig
import cv2
from torch.utils.data import Dataset, DataLoader
import gc

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
        text = text.lower()
        text = self.tokenizer(text, max_length=128, padding="max_length", truncation=True, return_tensors="pt")

        input_ids = text["input_ids"][0]
        attention_mask = text["attention_mask"][0]
        if self.augmentations:
            augmented = self.augmentations(image=image)
            image = augmented['image']
        if "label_group" in self.df.columns:
            return image, input_ids, attention_mask, torch.tensor(row.label_group)
        else:
            return image, input_ids, attention_mask


def pred_model(df, nlp_config_path, model_path, pred_name, net_module, config):
    print(f"pred_name: {pred_name} start")
    nlp_config = AutoConfig.from_pretrained(nlp_config_path)
    tokenizer = AutoTokenizer.from_pretrained(nlp_config_path,
                                              config=nlp_config)
    nlp_model = AutoModel.from_pretrained(nlp_config_path,
                                          config=nlp_config)

    model = net_module(config,
                       bert=nlp_model,
                       pretrained=False)
    model.load_state_dict(torch.load(model_path))
    model.to("cuda")
    print("model loaded")

    dataset = ShopeeDataset(df=df,
                            transforms=config.val_transforms,
                            tokenizer=tokenizer)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=16,
        num_workers=4,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )
    features = predict(model=model,
                       loader=loader,
                       device="cuda",
                       df=df,
                       pred_name=pred_name)
    del model
    del dataset
    del loader
    gc.collect()
    torch.cuda.empty_cache()
    return features


def get_kurupical_embeddings(model_path, df):
    model_dict = get_model(model_path)
    return pred_model(
           df=df,
           nlp_config_path=model_dict["nlp_config"],
           model_path=model_dict["model_path"],
           pred_name=model_dict["model_name"],
           net_module=model_dict["net_module"],
           config=model_dict["config"]
    )

def get_model(model_path):
    if "vit_residual_exp097.pth" in model_path:
        ret = {
            "nlp_config": "/kaggle/input/huggingface-bert/bert-base-multilingual-uncased",
            "model_path": model_path,
            "model_name": model_path,
            "net_module": exp097_3out.ShopeeNet,
            "config": exp097_3out.Config(linear_out=2048,
                                         dim=(384, 384),
                                         model_name="vit_base_patch16_384")
        }
        ret["config"].metric_layer_params = {
            "in_features": 2048,
            "out_features": 11014,
            "s": 32,
            "m": 0.5
        }

        return ret

    if "swin_large_exp103.pth" in model_path:
        ret = {
            "nlp_config": "/kaggle/input/cahya-distilbert-base-indonesian",
            "model_path": model_path,
            "model_name": model_path,
            "net_module": exp103_3out.ShopeeNet,
            "config": exp103_3out.Config(linear_out=2048,
                                         dim=(224, 224),
                                         model_name="swin_large_patch4_window7_224",
                                         nlp_model_name="distilbert")
        }
        ret["config"].metric_layer_params = {
            "in_features": 2048,
            "out_features": 11014,
            "s": 32,
            "m": 0.5
        }
        return ret

    if "swin_exp095.pth" in model_path:
        ret = {
            "nlp_config": "/kaggle/input/huggingface-bert/bert-base-multilingual-uncased",
            "model_path": model_path,
            "model_name": model_path,
            "net_module": exp095_3out.ShopeeNet,
            "config": exp095_3out.Config(linear_out=2048,
                                         dim=(384, 384),
                                         model_name="swin_base_patch4_window12_384")
        }
        ret["config"].metric_layer_params = {
            "in_features": 2048,
            "out_features": 11014,
            "s": 32,
            "m": 0.5
        }
        return ret

    if "vit_exp095.pth" in model_path:
        ret = {
            "nlp_config": "/kaggle/input/huggingface-bert/bert-base-multilingual-uncased",
            "model_path": model_path,
            "model_name": model_path,
            "net_module": exp095_3out.ShopeeNet,
            "config": exp095_3out.Config(linear_out=2048,
                                         dim=(384, 384),
                                         model_name="vit_base_patch16_384")
        }
        ret["config"].metric_layer_params = {
            "in_features": 2048,
            "out_features": 11014,
            "s": 32,
            "m": 0.5
        }
        return ret

    if "nfnet_l1_224_exp104.pth" in model_path:
        ret = {
            "nlp_config": "/kaggle/input/huggingface-bert/bert-base-multilingual-uncased",
            "model_path": model_path,
            "model_name": model_path,
            "net_module": exp104_3out.ShopeeNet,
            "config": exp104_3out.Config(linear_out=2048,
                                         dim=(224, 224),
                                         model_name="eca_nfnet_l1",
                                         nlp_model_name="bert"),
        }
        return ret

    if "swin_base_224_exp103.pth" in model_path:
        ret = {
            "nlp_config": "/kaggle/input/cahya-bert",
            "model_path": model_path,
            "model_name": model_path,
            "net_module": exp103_3out.ShopeeNet,
            "config": exp103_3out.Config(linear_out=2048,
                                         dim=(224, 224),
                                         model_name="swin_base_patch4_window7_224",
                                         nlp_model_name="bert"),
        }
        return ret

    if "vit_base_patch32_384_exp102.pth" in model_path:
        ret = {
            "nlp_config": "/kaggle/input/huggingface-bert/bert-base-multilingual-uncased",
            "model_path": model_path,
            "model_name": model_path,
            "net_module": exp102_3out.ShopeeNet,
            "config": exp102_3out.Config(linear_out=2048,
                                         dim=(384, 384),
                                         model_name="vit_base_patch32_384"),
        }
        return ret

    if "vit_base_patch16_224_exp104.pth" in model_path:
        ret = {
            "nlp_config": "/kaggle/input/huggingface-bert/bert-base-multilingual-uncased",
            "model_path": model_path,
            "model_name": model_path,
            "net_module": exp104_3out.ShopeeNet,
            "config": exp104_3out.Config(linear_out=2048,
                                         dim=(224, 224),
                                         model_name="vit_base_patch16_224",
                                         nlp_model_name="bert"),
        }
        return ret

    if "swin_large_roberta_exp112.pth" in model_path:
        import exp112
        ret = {
            "nlp_config": "/kaggle/input/xlm-roberta-base",
            "model_path": model_path,
            "model_name": model_path,
            "net_module": exp112.ShopeeNet,
            "config": exp112.Config(linear_out=2048,
                                    dim=(224, 224),
                                    model_name="swin_large_patch4_window7_224",
                                    nlp_model_name="xlm-roberta-base"),
        }
        return ret

    if "swin_large_bert_multi_exp112.pth" in model_path:
        import exp112
        ret = {
            "nlp_config": "/kaggle/input/huggingface-bert/bert-base-multilingual-uncased",
            "model_path": model_path,
            "model_name": model_path,
            "net_module": exp112.ShopeeNet,
            "config": exp112.Config(linear_out=2048,
                                    dim=(224, 224),
                                    model_name="swin_large_patch4_window7_224",
                                    nlp_model_name="bert-base-multilingual-uncased"),
        }
        return ret

    if "swin_base_xlm_roberta_exp113.pth" in model_path:
        import exp113
        ret = {
            "nlp_config": "/kaggle/input/xlm-roberta-base",
            "model_path": model_path,
            "model_name": model_path,
            "net_module": exp113.ShopeeNet,
            "config": exp113.Config(linear_out=2048,
                                    dim=(224, 224),
                                    model_name="swin_base_patch4_window7_224",
                                    nlp_model_name="xlm-roberta-base"),
        }
        return ret

    if "transformer_effv3_bertindonesian_exp114.pth" in model_path:
        import exp114
        ret = {
            "nlp_config": "/kaggle/input/cahya-bert",
            "model_path": model_path,
            "model_name": model_path,
            "net_module": exp114.ShopeeNet,
            "config": exp114.Config(linear_out=2048,
                                    dim=(224, 224),
                                    model_name="efficientnet_b3",
                                    nlp_model_name="bert_indonesian"),
        }
        return ret

    if "swin_large_roberta_exp133.pth" in model_path:
        import exp133
        ret = {
            "nlp_config": "/kaggle/input/xlm-roberta-base",
            "model_path": model_path,
            "model_name": model_path,
            "net_module": exp133.ShopeeNet,
            "config": exp133.Config(linear_out=2048,
                                    dim=(224, 224),
                                    model_name="swin_large_patch4_window7_224",
                                    nlp_model_name="xlm-roberta-base"),
        }
        return ret

    raise ValueError(f"model not found.{model_path}")

def predict(model, loader, device, df, pred_name):
    model.eval()

    img_features = []
    text_features = []
    all_features = []

    with torch.no_grad():
        for d in tqdm.tqdm(loader):
            images = d[0].to(device)
            input_ids = d[1].to(device)
            attention_mask = d[2].to(device)

            img, text, concat = model(images, input_ids, attention_mask)

            img_features.extend(img.detach().cpu().numpy().astype(np.float16))
            text_features.extend(text.detach().cpu().numpy().astype(np.float16))
            all_features.extend(concat.detach().cpu().numpy().astype(np.float16))

    img_features = np.array(img_features, dtype=np.float16)
    text_features = np.array(text_features, dtype=np.float16)
    all_features = np.array(all_features, dtype=np.float16)

    return img_features, text_features, all_features