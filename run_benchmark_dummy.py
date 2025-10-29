import os
import random
import time

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from utils.benchmark_utils import annotate, create_parser


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(10, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, 4096),
            torch.nn.ReLU(),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(),
            torch.nn.Linear(4096, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)

def set_rand_seeds(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(args):
    set_rand_seeds(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    model = Model().to(device)
    model.eval()

    input_tensor = torch.randn(8, 10, device=device)
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # warmup
    with torch.no_grad():
        for _ in range(3):
            model(input_tensor)

    # run inference 10 times and compute mean / variance
    timings = []
    last_output = None
    with torch.no_grad():
        for _ in range(10):
            if device.type == "cuda":
                torch.cuda.synchronize()
            begin = time.time()
            last_output = model(input_tensor)
            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.time()
            timings.append(end - begin)
    timings_tensor = torch.tensor(timings, device=device).cpu()
    mean_time = timings_tensor.mean().item()
    var_time = timings_tensor.var(unbiased=False).item()
    print("time mean/var:", timings_tensor.tolist(), mean_time, var_time)

    output_values = last_output.detach().cpu().flatten().tolist()
    with open(args.output_file, "w") as out_file:
        out_file.write("\n".join(str(v) for v in output_values))

    # optionally generate PyTorch Profiler trace
    # this is done after benchmarking because tracing introduces overhead
    if args.trace_file is not None:
        model.forward = annotate(model.forward, "forward_pass")
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            with record_function("timed_region"):
                with torch.no_grad():
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    model(input_tensor)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
        prof.export_chrome_trace(args.trace_file)


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
