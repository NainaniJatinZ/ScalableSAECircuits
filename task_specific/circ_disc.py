# %% 
import json
from sae_lens import SAE, HookedSAETransformer
from functools import partial
import einops
import os
import gc
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
from datasets import load_dataset
from transformers import AutoTokenizer
from transformer_lens.hook_points import (
    HookPoint,
) 
import numpy as np
import pandas as pd
from pprint import pprint as pp
from typing import Tuple
from torch import Tensor
from functools import lru_cache
from typing import TypedDict, Optional, Tuple, Union
from tqdm import tqdm
import random
import helpers.utils as utils
from importlib import reload

# %%
with open("config.json", 'r') as file:
    config = json.load(file)
token = config.get('huggingface_token', None)
os.environ["HF_TOKEN"] = token

# Define device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

hf_cache = "/work/pi_jensen_umass_edu/jnainani_umass_edu/mechinterp/huggingface_cache/hub"
os.environ["HF_HOME"] = hf_cache

# Load the model
model = HookedSAETransformer.from_pretrained("google/gemma-2-9b", device=device, cache_dir=hf_cache) 

pad_token_id = model.tokenizer.pad_token_id
for param in model.parameters():
    param.requires_grad_(False)

# %%

from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory

# TODO: Make this nicer.
df = pd.DataFrame.from_records({k:v.__dict__ for k,v in get_pretrained_saes_directory().items()}).T
df.drop(columns=["expected_var_explained", "expected_l0", "config_overrides", "conversion_func"], inplace=True)
df[df['model']=='gemma-2-9b'] # Each row is a "release" which has multiple SAEs which may have different configs / match different hook points in a model. 
# %%
import re
from collections import defaultdict

neuronpedia_dict = df.loc['gemma-scope-9b-pt-res']['neuronpedia_id']
# Dictionary to store the closest string for each layer
closest_strings = {}

# Regular expression to extract the layer number and l0 value
pattern = re.compile(r'layer_(\d+)/width_16k/average_l0_(\d+)')

# Organize strings by layer
layer_dict = defaultdict(list)

for s in neuronpedia_dict.keys():
    match = pattern.search(s)
    if match and neuronpedia_dict[s] != None:
        layer = int(match.group(1))
        l0_value = int(match.group(2))
        layer_dict[layer].append((s, l0_value))

# Find the string with l0 value closest to 100 for each layer
for layer, items in layer_dict.items():
    closest_string = min(items, key=lambda x: abs(x[1] - 32))
    closest_strings[layer] = closest_string[0]
closest_strings

# %%

device = "cuda"
sae_gap = 13
layers = [i for i in range(0, model.cfg.n_layers, sae_gap)]
saes = [
    SAE.from_pretrained(
        release="gemma-scope-9b-pt-res",
        sae_id=closest_strings[layer],
        device=str(device)
    )[0]
    for layer in tqdm(layers)
]

# %%
import json

task = "sva/rc_train"
file_path = 'data/'+task+'.json'
example_length = 7
with open(file_path, 'r') as file:
    data = [json.loads(line) for line in file]
for entry in data:
    print(entry)
    break

# %%

clean_data = []
corr_data = []
clean_labels = []
corr_labels = []
for entry in data:
    clean_len = len(model.tokenizer(entry['clean_prefix']).input_ids)
    corr_len = len(model.tokenizer(entry['patch_prefix']).input_ids)
    if clean_len == corr_len == example_length:
        clean_data.append(entry['clean_prefix'])
        corr_data.append(entry['patch_prefix'])
        clean_labels.append(entry['clean_answer'])
        corr_labels.append(entry['patch_answer'])
print(len(clean_data))

N = 3000
clean_tokens = model.to_tokens(clean_data[:N])
corr_tokens = model.to_tokens(corr_data[:N])
clean_label_tokens = model.to_tokens(clean_labels[:N], prepend_bos=False).squeeze(-1)
corr_label_tokens = model.to_tokens(corr_labels[:N], prepend_bos=False).squeeze(-1)
print(clean_tokens.shape, corr_tokens.shape)

def logit_diff_fn(logits, clean_labels, corr_labels, token_wise=False):
    clean_logits = logits[torch.arange(logits.shape[0]), -1, clean_labels]
    corr_logits = logits[torch.arange(logits.shape[0]), -1, corr_labels]
    return (clean_logits - corr_logits).mean() if not token_wise else (clean_logits - corr_logits)

batch_size = 16 
clean_tokens = clean_tokens[:batch_size*(len(clean_tokens)//batch_size)]
corr_tokens = corr_tokens[:batch_size*(len(corr_tokens)//batch_size)]
clean_label_tokens = clean_label_tokens[:batch_size*(len(clean_label_tokens)//batch_size)]
corr_label_tokens = corr_label_tokens[:batch_size*(len(corr_label_tokens)//batch_size)]

clean_tokens = clean_tokens.reshape(-1, batch_size, clean_tokens.shape[-1])
corr_tokens = corr_tokens.reshape(-1, batch_size, corr_tokens.shape[-1])
clean_label_tokens = clean_label_tokens.reshape(-1, batch_size)
corr_label_tokens = corr_label_tokens.reshape(-1, batch_size)

print(clean_tokens.shape, corr_tokens.shape, clean_label_tokens.shape, corr_label_tokens.shape)

# %%
use_mask = False 
mean_mask = False
avg_logit_diff = 0
utils.cleanup_cuda()
with torch.no_grad():
    for i in range(10):
        logits = model(
            clean_tokens[i]
            )
        ld = logit_diff_fn(logits, clean_label_tokens[i], corr_label_tokens[i])
        print(ld)
        avg_logit_diff += ld
        del logits
        utils.cleanup_cuda()
model.reset_hooks(including_permanent=True)
model.reset_saes()
avg_model_diff = (avg_logit_diff / 10).item()
print("Average Full Model LD: ", avg_model_diff)

# %%
use_mask = False 
mean_mask = False
avg_logit_diff = 0
utils.cleanup_cuda()
with torch.no_grad():
    for i in range(10):
        logits, saes = utils.run_sae_hook_fn(model, saes, clean_tokens[i])
        ld = logit_diff_fn(logits, clean_label_tokens[i], corr_label_tokens[i])
        print(ld)
        avg_logit_diff += ld
        del logits
        utils.cleanup_cuda()
model.reset_hooks(including_permanent=True)
model.reset_saes()
avg_logit_diff = (avg_logit_diff / 10).item()
print("Average Model + Saes LD: ", avg_logit_diff)

# %%

for sae in saes:
    sae.mask = utils.SparseMask(sae.cfg.d_sae, 1.0, seq_len=example_length).to(device)

saes = utils.get_sae_means(model, saes, corr_tokens, 40, 16)

saes = utils.get_sae_error_means(model, saes, corr_tokens, 40, 16)

# %%

use_mask = False 
mean_mask = False
avg_logit_err_diff = 0
utils.cleanup_cuda()
with torch.no_grad():
    for i in range(10):
        logits, saes = utils.run_sae_hook_fn(model, saes, clean_tokens[i], use_mean_error=True)
        ld = logit_diff_fn(logits, clean_label_tokens[i], corr_label_tokens[i])
        print(ld)
        avg_logit_err_diff += ld
        del logits
        utils.cleanup_cuda()
model.reset_hooks(including_permanent=True)
model.reset_saes()
avg_logit_err_diff = (avg_logit_err_diff / 10).item()
print("Average Model + Saes (Mean Error) LD: ", avg_logit_err_diff)

# %%

thresholds = []
modify_fn=lambda x: x**2
start_threshold = 0.01
end_threshold = 0.2
n_runs = 5
delta = (end_threshold - start_threshold) / n_runs
def linear_map(x):
        mod_start = modify_fn(start_threshold)
        mod_end = modify_fn(end_threshold)
        return (x - mod_start) / (mod_end - mod_start) * (end_threshold - start_threshold) + start_threshold
    
mf = lambda x: linear_map(modify_fn(x))
for i in range(n_runs):
    thresholds.append(
        mf(start_threshold + i*delta)
        )
thresholds

# %%
for i in thresholds:
    utils.do_training_run(model, saes, token_dataset=clean_tokens, labels_dataset= clean_label_tokens, corr_labels_dataset=corr_label_tokens, sparsity_multiplier=i, task=task+"w_error", example_length=example_length, loss_function="logit_diff", per_token_mask=True, use_mask=True, mean_mask=True, portion_of_data=0.3, use_mean_error=True)

# %%
