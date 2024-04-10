import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import torch
import torch.nn.functional as F
from transformers import pipeline

classifier = pipeline("sentiment-analysis")

while True:
    INPUT = input("> ")
    res = classifier(INPUT)
    print(res)