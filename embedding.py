import torch
import numpy as np
from transformers import AutoProcessor, AutoModel
from datasets import load_dataset
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
model = AutoModel.from_pretrained("microsoft/layoutlmv3-base")
model.to(device)

dataset = load_dataset("nielsr/funsd-layoutlmv3", split="train")
# dataloader = DataLoader(dataset, batch_size=1)

sample = dataset[0]
image = sample["image"]
image = np.array(image)

for sample in dataset:
    image = sample["image"]
    words = sample["tokens"]
    boxes = sample["bboxes"]
    encoding = processor(image, words, boxes=boxes, return_tensors="pt")
    encoding["bbox"] = encoding["bbox"][:,:512]
    encoding["input_ids"] = encoding["input_ids"][:,:512]
    encoding["attention_mask"] = encoding["attention_mask"][:,:512]
    encoding.to(device)
    outputs = model(**encoding)
    last_hidden_states = outputs.last_hidden_state
    print(last_hidden_states.shape)
