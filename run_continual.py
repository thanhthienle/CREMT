import yaml
import torch
from config import Param
from methods.utils import setup_seed
from methods.manager import Manager, NashManager


def run(args):
    print(f"Hyper-parameter configurations:")
    print(yaml.dump(args.__dict__, sort_keys=True, indent=4))
    with open(f"live_{args.logname}", "a") as writer:
        writer.write(yaml.dump(args.__dict__, sort_keys=True, indent=4))

    setup_seed(args.seed)
    if args.mtl == "nashmtl":
        manager = NashManager(args)
    else: manager = Manager(args)
    manager.train(args)


if __name__ == "__main__":
    # Load configuration
    param = Param()
    args = param.args

    # Device
    torch.cuda.set_device(args.gpu)
    args.device = torch.device(args.device)

    # Num GPU
    args.n_gpu = torch.cuda.device_count()

    # Task name
    args.task_name = args.dataname

    # rel_per_task
    args.rel_per_task = 8 if args.dataname == "FewRel" else 4

    # Run
    run(args)
