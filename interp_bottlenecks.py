import os
import json
import argparse
import torch
import numpy as np
import pandas as pd
from functools import partial
from tqdm import tqdm
from collections import defaultdict
import re
from sae_lens import SAE, HookedSAETransformer
from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory
import helpers.utils as utils

##################################
# Utility Functions
##################################

def logit_diff_fn(logits, clean_labels, corr_labels, token_wise=False):
    """
    Compute logit difference for a batch.
    """
    # logits shape: [batch, seq_len, vocab_size]
    clean_logits = logits[torch.arange(logits.shape[0]), -1, clean_labels]
    corr_logits = logits[torch.arange(logits.shape[0]), -1, corr_labels]
    if token_wise:
        return (clean_logits - corr_logits)
    else:
        return (clean_logits - corr_logits).mean()

def parse_args():
    """
    Set up argument parsing for controlling:
    - `task` name
    - `sae_gap`
    - `example_length`
    - `num_batches` (or how many chunks to use)
    - `batch_size`
    - `use_mask`, `use_mean_error`, etc.
    - thresholds for the training step
    """
    parser = argparse.ArgumentParser(
        description="Run a systematic script for hooking in SAEs, computing logit diffs, and optionally doing training."
    )

    # Basic parameters
    parser.add_argument("--config_file", type=str, default="config.json",
                        help="Path to a JSON config file with optional HF token.")
    parser.add_argument("--hf_cache", type=str, default="/work/pi_jensen_umass_edu/jnainani_umass_edu/mechinterp/huggingface_cache/hub",
                        help="Path to the Hugging Face cache directory.")
    parser.add_argument("--task", type=str, default="sva/rc_train",
                        help="Task subfolder + file name (like 'sva/rc_train'). Will look in data/<task>.json.")
    parser.add_argument("--example_length", type=int, default=7,
                        help="Expected length of the tokenized example (used for filtering).")
    parser.add_argument("--N", type=int, default=3000,
                        help="How many samples to load from the dataset (max).")

    # Model / SAE parameters
    parser.add_argument("--model_name", type=str, default="google/gemma-2-9b",
                        help="Which model to load with HookedSAETransformer.")
    parser.add_argument("--sae_gap", type=int, default=13,
                        help="Which gap to use for collecting SAEs. (E.g. load every 13th layer).")
    parser.add_argument("--layer_l0_target", type=int, default=100,
                        help="Which L0 value is the target or closest choice for each layer's SAE.")
    parser.add_argument("--num_batches_eval", type=int, default=10,
                        help="How many batches to evaluate on when computing average logit diff.")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size to use for chunked inference or training.")
    
    # Mask/mean usage
    parser.add_argument("--use_mask", action="store_true",
                        help="Whether to use an SAE mask at inference/training time.")
    parser.add_argument("--mean_mask", action="store_true",
                        help="Whether to take the average of the mask across tokens (per-token mask).")
    parser.add_argument("--use_mean_error", action="store_true",
                        help="Whether to use the error means for the SAE activation instead of the plain mean.")
    parser.add_argument("--per_token_mask", action="store_true")
    
    # For the training runs with thresholds
    parser.add_argument("--run_training_thresholds", action="store_true",
                        help="Whether to run the threshold-based training loop.")
    parser.add_argument("--start_threshold", type=float, default=0.01,
                        help="Lower bound for threshold-based training.")
    parser.add_argument("--end_threshold", type=float, default=0.2,
                        help="Upper bound for threshold-based training.")
    parser.add_argument("--n_threshold_steps", type=int, default=5,
                        help="Number of steps between start_threshold and end_threshold (inclusive).")
    parser.add_argument("--portion_of_data", type=float, default=0.3,
                        help="Portion of data used in the training run for thresholds.")

    return parser.parse_args()


def main():
    ##############################
    # Parse Command Line Args
    ##############################
    args = parse_args()

    # Potentially load configuration from JSON
    if os.path.exists(args.config_file):
        with open(args.config_file, 'r') as file:
            config = json.load(file)
        token = config.get('huggingface_token', None)
        if token is not None:
            os.environ["HF_TOKEN"] = token
    else:
        print(f"WARNING: config file '{args.config_file}' not found. Continuing without it.")
        token = None

    # Set HF cache environment variable if provided
    if args.hf_cache:
        os.environ["HF_HOME"] = args.hf_cache

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    ##############################
    # Load Model
    ##############################
    model = HookedSAETransformer.from_pretrained(
        args.model_name,
        device=device,
        cache_dir=args.hf_cache
    )

    # Make sure to freeze parameters
    for param in model.parameters():
        param.requires_grad_(False)

    # Retrieve any needed info from model
    pad_token_id = model.tokenizer.pad_token_id

    ##############################
    # Load SAEs (Nearest L0)
    ##############################
    df = pd.DataFrame.from_records({k:v.__dict__ for k,v in get_pretrained_saes_directory().items()}).T
    df.drop(columns=["expected_var_explained", "expected_l0", "config_overrides", "conversion_func"], inplace=True)
    # # This is your custom approach from the notebook:
    if args.model_name == "google/gemma-2-9b":
        release_name = "gemma-scope-9b-pt-res"
    elif args.model_name == "google/gemma-2-2b":
        release_name = "gemma-scope-2b-pt-res"
    neuronpedia_dict = df.loc[release_name]['saes_map']
    pattern = re.compile(r'layer_(\d+)/width_16k/average_l0_(\d+)')
    layer_dict = defaultdict(list)
    for s in neuronpedia_dict.keys():
        match = pattern.search(s)
        if match and neuronpedia_dict[s] is not None:
            layer = int(match.group(1))
            l0_value = int(match.group(2))
            layer_dict[layer].append((s, l0_value))

    # Find the string with l0 value closest to the user-specified target
    closest_strings = {}
    for layer, items in layer_dict.items():
        closest_string = min(items, key=lambda x: abs(x[1] - args.layer_l0_target))
        closest_strings[layer] = closest_string[0]

    # Actually load the SAEs
    layers = [i for i in range(0, model.cfg.n_layers, args.sae_gap)]
    saes = []
    for layer in tqdm(layers):
        sae_id = closest_strings.get(layer, None)
        if sae_id is not None:
            sae = SAE.from_pretrained(
                release=release_name,
                sae_id=sae_id,
                device=device
            )[0]
            saes.append(sae)
        else:
            print(f"Warning: No matching SAE ID found for layer {layer} with L0 target {args.layer_l0_target}")

    ##############################
    # Load Data
    ##############################
    # Example: data/<task>.json
    data_path = f"data/{args.task}.json"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file {data_path} not found.")

    with open(data_path, 'r') as f:
        data = [json.loads(line) for line in f]

    # Some sample checks
    if not data:
        raise ValueError("Loaded data is empty!")

    # Filter dataset by example_length
    clean_data = []
    corr_data = []
    clean_labels = []
    corr_labels = []
    
    for entry in data:
        clean_len = len(model.tokenizer(entry['clean_prefix']).input_ids)
        corr_len = len(model.tokenizer(entry['patch_prefix']).input_ids)
        if clean_len == corr_len == args.example_length:
            clean_data.append(entry['clean_prefix'])
            corr_data.append(entry['patch_prefix'])
            clean_labels.append(entry['clean_answer'])
            corr_labels.append(entry['patch_answer'])
    
    # Limit to top N
    clean_data = clean_data[:args.N]
    corr_data = corr_data[:args.N]
    clean_labels = clean_labels[:args.N]
    corr_labels = corr_labels[:args.N]

    # Tokenize
    clean_tokens = model.to_tokens(clean_data)
    corr_tokens = model.to_tokens(corr_data)
    clean_label_tokens = model.to_tokens(clean_labels, prepend_bos=False).squeeze(-1)
    corr_label_tokens = model.to_tokens(corr_labels, prepend_bos=False).squeeze(-1)

    # Reshape into batches
    n_batches_total = (len(clean_tokens) // args.batch_size)
    clean_tokens = clean_tokens[:n_batches_total*args.batch_size]
    corr_tokens = corr_tokens[:n_batches_total*args.batch_size]
    clean_label_tokens = clean_label_tokens[:n_batches_total*args.batch_size]
    corr_label_tokens = corr_label_tokens[:n_batches_total*args.batch_size]

    clean_tokens = clean_tokens.reshape(-1, args.batch_size, clean_tokens.shape[-1])
    corr_tokens = corr_tokens.reshape(-1, args.batch_size, corr_tokens.shape[-1])
    clean_label_tokens = clean_label_tokens.reshape(-1, args.batch_size)
    corr_label_tokens = corr_label_tokens.reshape(-1, args.batch_size)

    print("Number of total batches after reshape:", clean_tokens.shape[0])

    ####################################
    # Evaluate full model logit diff
    ####################################
    # (This is just an example demonstration.)

    print("Computing average full model logit diff (clean vs corr)...")
    avg_model_diff = 0.0
    # utils.cleanup_cuda()
    with torch.no_grad():
        for i in range(min(args.num_batches_eval, clean_tokens.shape[0])):
            logits = model(clean_tokens[i])  # shape [batch_size, seq_len, vocab_size]
            ld = logit_diff_fn(logits, clean_label_tokens[i], corr_label_tokens[i])
            avg_model_diff += ld
    avg_model_diff = (avg_model_diff / min(args.num_batches_eval, clean_tokens.shape[0])).item()
    print("Average Full Model LD:", avg_model_diff)

    ####################################
    # Evaluate model + SAEs
    ####################################
    # # If you want to run the model with SAEs directly:
    # # E.g., using a custom utility like `utils.run_sae_hook_fn(model, saes, ...)`.
    # # reset hooks if your library requires it
    # model.reset_hooks(including_permanent=True)
    # model.reset_saes()

    avg_logit_diff = 0.0
    with torch.no_grad():
        for i in range(min(args.num_batches_eval, clean_tokens.shape[0])):
            logits, saes = utils.run_sae_hook_fn(model, saes, clean_tokens[i], use_mean_error=False)
            ld = logit_diff_fn(logits, clean_label_tokens[i], corr_label_tokens[i])
            avg_logit_diff += ld
    avg_logit_diff = (avg_logit_diff / min(args.num_batches_eval, clean_tokens.shape[0])).item()
    print("Average Model + SAEs LD:", avg_logit_diff)

    ####################################
    # Compute means & error means for SAEs
    ####################################
    if args.use_mask:
        # Example usage of a custom utility to create a mask
        for sae in saes:
            sae.mask = utils.SparseMask(sae.cfg.d_sae, 1.0, seq_len=args.example_length).to(device)
    
    saes = utils.get_sae_means(model, saes, corr_tokens, total_batches=40, batch_size=16)
    saes = utils.get_sae_error_means(model, saes, corr_tokens, total_batches=40, batch_size=16)

    ####################################
    # Evaluate model + SAEs with error means
    ####################################
    # # reset hooks again
    # model.reset_hooks(including_permanent=True)
    # model.reset_saes()
    save_task = args.task + "_saegap" + str(args.sae_gap)
    if args.use_mean_error:
        save_task += "_mean_error"
        avg_logit_err_diff = 0.0
        with torch.no_grad():
            for i in range(min(args.num_batches_eval, clean_tokens.shape[0])):
                logits, saes = utils.run_sae_hook_fn(
                    model, saes, clean_tokens[i], use_mean_error=True
                )
                ld = logit_diff_fn(logits, clean_label_tokens[i], corr_label_tokens[i])
                avg_logit_err_diff += ld
        avg_logit_err_diff = (avg_logit_err_diff / min(args.num_batches_eval, clean_tokens.shape[0])).item()
        print("Average Model + SAEs (Mean Error) LD:", avg_logit_err_diff)

    ####################################
    # Optionally do training with thresholds
    ####################################
    if args.run_training_thresholds:
        # For example, you can create a list of thresholds
        def modify_fn(x):
            return x**2
        
        def linear_map(x, start, end):
            mod_start = modify_fn(start)
            mod_end = modify_fn(end)
            return (x - mod_start) / (mod_end - mod_start) * (end - start) + start

        # Create thresholds from start_threshold to end_threshold in n_threshold_steps
        thresholds = []
        n_runs = args.n_threshold_steps
        delta = (args.end_threshold - args.start_threshold) / float(n_runs)
        for i in range(n_runs):
            thresholds.append(linear_map(modify_fn(args.start_threshold + i*delta),args.start_threshold,args.end_threshold))
            # thresholds.append(args.start_threshold + i*delta)

        print("Thresholds to run training on:", thresholds)

        # Example usage of a custom training utility:
        for thr in thresholds:
            utils.do_training_run(
                model=model,
                saes=saes,
                token_dataset=clean_tokens,
                labels_dataset=clean_label_tokens,
                corr_labels_dataset=corr_label_tokens,
                sparsity_multiplier=thr,
                task=save_task,
                example_length=args.example_length,
                loss_function="logit_diff",
                per_token_mask=args.per_token_mask,
                use_mask=args.use_mask,
                mean_mask=args.mean_mask,
                portion_of_data=args.portion_of_data,
                use_mean_error=args.use_mean_error
            )
            print(f"Training run complete for threshold={thr}")

    print("Script finished successfully.")

if __name__ == "__main__":
    main()