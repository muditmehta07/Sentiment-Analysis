import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import torch
import nltk
import jsonlines

class ProcessData:
    def __init__(self):
        self.dataset_dir = "./dataset"
        self.initial_dataset = f"{self.dataset_dir}/data.jsonl"
        self.train_dataset = f"{self.dataset_dir}/train.jsonl"
        self.test_dataset = f"{self.dataset_dir}/test.jsonl"
        self.validation_dataset = f"{self.dataset_dir}/validation.jsonl"

    def print_dataset(self, dataset):
        with jsonlines.open(dataset) as f:
            for i in f:
                print(i)

p = ProcessData()
p.print_dataset(p.initial_dataset)
