# util/model_loader.py

import os
import torch

# ─── pick device: CUDA → CPU ────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[model_loader] using device: {device}")

def get_model(args, num_classes, load_ckpt=True):
    if args.model_arch == 'fno1d':
        from nn_FNO import FNO1d
        from nn_step_methods import RK4step

        net = FNO1d(args.modes,
                    args.width,
                    args.time_future,
                    args.time_history)

        if load_ckpt:
            ckpt = torch.load(args.net_file, map_location=device)
            net.load_state_dict(ckpt)

        class FNO1dStepper(torch.nn.Module):
            def __init__(self, fno, dt):
                super().__init__()
                self.fno = fno
                self.dt  = dt

            def forward(self, x):
                if x.dim() == 2:
                    x = x.unsqueeze(-1)
                out = RK4step(self.fno, x, self.dt)
                return out.squeeze(-1)

            def forward_threshold(self, x, threshold):
                y = self.forward(x)
                return torch.clamp(y, max=threshold)

        model = FNO1dStepper(net, args.time_step)
        return model.to(device).eval()

    # … your other branches (resnet/mobilenet) unchanged, just call .to(device) …
    raise ValueError(f"Unsupported model_arch: {args.model_arch}")
