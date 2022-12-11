import torch
from torch.utils.data import DataLoader

from transformers import AutoModel
from transforms import img_train_transform
from funsd_data import FUNSD

funsd_data = FUNSD("../dataset/training_data/")
dataloader = DataLoader(funsd_data, batch_size=1)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModel.from_pretrained("microsoft/layoutlmv3-base")
model.to(device)

for encoding in dataloader:
    encoding.to(device)
    outputs = model(**encoding)
    last_hidden_states = outputs.last_hidden_state
    print(last_hidden_states.shape)
