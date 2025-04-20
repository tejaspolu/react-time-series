#!/usr/bin/env python3
import os
import numpy as np
import torch
from tqdm import tqdm

from util.args_loader import get_args
from util.data_loader  import get_loader_in
from util.model_loader import get_model

# pick device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[compute_threshold] running on {device}\n")

# parse args & seeds
args = get_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
torch.manual_seed(1)
if device.type == "cuda":
    torch.cuda.manual_seed_all(1)
np.random.seed(1)

# hook helper for CNNs
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def eval_ood_detector(args):
    out_dir = os.path.join(args.base_dir, args.in_dataset, args.method, args.name)
    os.makedirs(out_dir, exist_ok=True)

    loader_in = get_loader_in(args, split=('val',))
    loader, _ = loader_in.val_loader, loader_in.num_classes

    model = get_model(args, None, load_ckpt=True)
    model.eval()

    activation_log = []
    seen = 0
    limit = args.lim

    pbar = tqdm(loader, desc="Threshold Estimation", unit="batch")
    for inputs, _ in pbar:
        if seen >= limit:
            break

        inputs = inputs.to(device)
        with torch.no_grad():
            if args.model_arch == 'fno1d':
                # RAW forward (no clamp)
                feats = model(inputs)        # (B, seq_len)
                arr   = feats.cpu().numpy()
            else:
                # CNN raw forward + avgpool hook
                handles = []
                if hasattr(model, 'avgpool'):
                    handles.append(model.avgpool.register_forward_hook(
                        get_activation('feat')
                    ))
                _ = model(inputs)
                for h in handles:
                    h.remove()
                feat = activation['feat']   # (B, C, H, W) or (B, C, L)
                a = feat.cpu().numpy()
                B, C = a.shape[0], a.shape[1]
                arr = a.reshape(B, C, -1).mean(axis=2)

            activation_log.append(arr)

        seen += inputs.size(0)
        pbar.set_postfix_str(f"seen={seen}")

    all_feats = np.vstack(activation_log)
    thr = np.percentile(all_feats.flatten(), 90)
    print(f"\n--> THRESHOLD at 90th percentile: {thr:.4f}")

if __name__ == '__main__':
    eval_ood_detector(args)
