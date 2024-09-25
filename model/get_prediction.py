import argparse
import pandas as pd

import os
import re
import zipfile
import importlib

import gdown
import numpy as np
import torch
from natsort import natsorted
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import gc
import torch
import torch
import torch.nn.functional as F
from torchvision import transforms
from tqdm.auto import trange
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from  torch import nn
import math
import itertools
import torchvision
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init
from torch import nn
from torch import distributions
from torch.distributions import MultivariateNormal, Uniform, TransformedDistribution, SigmoidTransform
from torch.nn.parameter import Parameter
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score


def transforms(df, train=True):
    text_fields = ['Задача en', 'Обстановка en', 'Оптимальный план en', 'Предсказанный план']
    emb = [model.encode(df[i]) for i in text_fields]
    new_feature_1 = np.array([util.dot_score(emb[0][i], emb[-1][i]) for i in range(df.shape[0])])
    new_feature_1 = np.squeeze(new_feature_1)
    X = np.hstack(emb)
    X = np.column_stack((X, new_feature_1))
    y = df['Успех предсказанного плана'] if train else None
    return X, y

def predict(df):
    X_test, _ = transforms(df, False)
    model = CatBoostClassifier()
    model.load_model('model.cb')
    return model.predict(X_test)

def main(path_to_csv):
    df = pd.read_csv(path_to_csv)
    predictions = predict(df)
    predictions_df = pd.DataFrame(predictions, index=df.index)
    predictions_df.to_csv('labels_123.csv', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="Path to the CSV file")
    args = parser.parse_args()

    main(args.path)