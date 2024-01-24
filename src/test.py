import glob
import json
import os
import numpy as np
import random
import torch
from torch import nn
import sys

import torch.distributed as dist
from torch.utils.data.dataloader import default_collate


from trainval_single_batch import downstream_task_forward
from loss import LabelSmoothingCrossEntropy

dataset_mapping = {
    'radvqa': 'datasets.medvqa_features',
    'pathvqa': 'datasets.medvqa_features',
    'slakevqa': 'datasets.medvqa_features',
}

def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def log(message, logger=None):
    """
    Placeholder log function; replace with your loggers/apis of choice (e.g. wandb, etc.)
    """
    if logger is not None: raise NotImplemented("implement your own logger")
    print(message)
    if args.freeze:
        args.log_path = os.path.join("./checkpoints", f"{args.dataset}", args.method)
    else:
        args.log_path = os.path.join("./checkpoints", f"{args.dataset}", args.method+f'_top{args.top_k}')
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    with open(os.path.join(args.log_path,  'log.txt'), 'a') as lf:
        lf.write(message+'\n')

def process_batch(batch, set_to_device=None, replace_empty_with_none=False):
    if set_to_device is not None:
        if isinstance(batch, dict):
            for key in batch:
                if torch.is_tensor(batch[key]):
                    batch[key] = batch[key].to(set_to_device)
        elif isinstance(batch, list):
            batch = [_b.to(set_to_device) if torch.is_tensor(_b) else _b for _b in batch]

    if replace_empty_with_none:
        if isinstance(batch, dict):
            for key in batch:
                if len(batch[key]) == 0:
                    batch[key] = None
        elif isinstance(batch, list):
            batch = [_b if len(_b) > 0 else None for _b in batch]

    return batch


def main(args):
    if not dist.is_nccl_available():
        print("Error: nccl backend not available.")
        sys.exit(1)

    seed_everything(args.seed)
    from model_memory import ModelConfig, VQAModel

    config = ModelConfig.from_args(args)
    device = torch.device("cuda")

    model = VQAModel(config, device).to(device)
    if args.checkpoint is not None:
        checkpoints = torch.load(args.checkpoint, map_location='cpu')['state_dict']
        try:
            model.load_state_dict(state_dict=checkpoints, strict=True)
        except:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in checkpoints.items():
                name = k.replace('module.', '')
                new_state_dict[name] = v
            model.load_state_dict(state_dict=new_state_dict, strict=True)


    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Total parameters: ", total_params)
    print("Trainable parameters: ", trainable_params)

    if args.dataset in dataset_mapping:
        module_name = dataset_mapping[args.dataset]
        VQADataset = getattr(__import__(module_name, fromlist=['VQADataset']),
                                       'VQADataset')
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}")


    dset_val = VQADataset(args, split="test")
    dldr_val = torch.utils.data.DataLoader(dset_val,
                                            batch_size=args.batch_size,
                                            shuffle=False,
                                            pin_memory=True,
                                            drop_last=False,
                                            num_workers=args.num_workers,
                                            collate_fn=default_collate)


    criterion = LabelSmoothingCrossEntropy()

    temp_result_dict = []
    log('Start evaluation')
    model.eval()

    count_all = 0
    count_true = 0
    for i, batch in enumerate(dldr_val):
        with torch.no_grad():
            batch = process_batch(batch, set_to_device=device, replace_empty_with_none=True)
            loss, call, ctrue = downstream_task_forward(model, batch, criterion, args, dset_val, temp_result_dict)
            count_all += call
            count_true += ctrue


    overall_acc = count_true/count_all
    log(f"overall_acc = {overall_acc}")
    with open(f'{args.log_path}/result.json', 'w') as f:
        json.dump(temp_result_dict, f, indent=4)




if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parser for LGVA training script.")

    # Training hyperparameters
    parser.add_argument('--batch_size', default=256, type=int, help="batch size")
    parser.add_argument('--lr', default=1e-4, type=float, help="learning rate")
    parser.add_argument('--wd', default=1e-2, type=float, help="weight decay")
    parser.add_argument('--epochs', default=50, type=int, help="number of training epochs")
    parser.add_argument('--grad_clip_val', default=1.0, type=float, help="gradient clip, must be set > 0 to enable")
    parser.add_argument('--seed', default=3407, type=int, help="random seed")
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--checkpoint', default=None, type=str)

    parser.add_argument('--gpus', default=1, type=int)  # NOTE: current script is set-up for single-gpu training only.
    parser.add_argument('--num_workers', default=1, type=int, help="number of dataset workers")

    # Model hyperparameters (for more help/details, see ModelConfig)
    parser.add_argument('--d_input', default=768, type=int, help="see ModelConfig")
    parser.add_argument('--d_output', default=512, type=int, help="see ModelConfig")
    parser.add_argument('--compressed_size', default=32, type=int, help="see ModelConfig")
    parser.add_argument('--top_k', default=5, type=int, help="see ModelConfig")
    parser.add_argument('--n_ca_heads', default=12, type=int, help="see ModelConfig")
    parser.add_argument('--ca_dropout', default=0.1, type=float, help="see ModelConfig")
    parser.add_argument('--method', type=str)

    # I/O and tools parameters
    parser.add_argument('--anno_path', type=str, help='Annotation', default='./data/Annotations/VQA-RAD')
    parser.add_argument('--data_path', type=str, help='Feature', default='./data/VQA-RAD')
    parser.add_argument('--memo_path', type=str, help='Memory', default='./data/VQA-RAD')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--visible', action='store_true') #for check each question predicts answer
    args = parser.parse_args()

    main(args)
