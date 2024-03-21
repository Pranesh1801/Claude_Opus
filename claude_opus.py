# -*- coding: utf-8 -*-
!pip install anthropic rich pytz

from anthropic import Anthropic
from rich import print
from datetime import datetime, timezone, timedelta
import re, json, os, requests, pytz

os.environ["ANTHROPIC_API_KEY"] = "Your-API-Key"

client = Anthropic()
MODEL_NAME = "Model and version of Claude (Eg. claude-3-opus-20240229)"

road_condition_message = {
    "role": "user",
    "content": "I am working on a project where I aim to train a PyTorch model on multiple GPUs. My input data is stored in separate files for each training example, and during preprocessing, I save them using the torch.save method to .pt files. Later, I load these files using DataLoader, where I want to set num_workers > 0 to speed up the process. However, it seems that num_workers can only be set to >0 when the input data is on CPU. My question is: Should I save CUDA tensors already and just use num_workers=0, or should I store CPU tensors, set num_workers > 0, and then move the batch as a whole to GPU? I'm uncertain which approach would be more efficient for training speed (time) on multiple GPUs. Any insights or best practices on this matter would be greatly appreciated."
}

message = client.messages.create(
    model=MODEL_NAME,
    max_tokens=1024,
    messages=[road_condition_message]
).content[0].text

print("##### Response w/o Function Calling #####\n\n" + message)
