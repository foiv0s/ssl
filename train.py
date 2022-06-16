import os
import argparse

import torch
from logger import create_logger
from utils import bool_flag
from tools.datasets import build_dataset, get_dataset, get_encoder_size, get_loss_type
from model import Model
from tools.checkpoint import Checkpointer
from task_self_supervised import train_self_supervised
from task_classifiers import train_classifiers

parser = argparse.ArgumentParser(description='Self-Labelling Siamese Networks')
# parameters for general training stuff
# --nmb_workers 0 --dataset c10 --loss_type 2 --encoder_mom 0.99 --lam 1 --size_crops 28

parser.add_argument('--dataset', type=str, default='C10')
parser.add_argument('--data_path', type=str, default='default_run', help='Path of dataset')
parser.add_argument('--nmb_workers', type=int, default=8, help='Number of workers on Transformation process')
parser.add_argument("--nmb_crops", type=int, default=[2], nargs="+",
                    help="list of number of crops (example: [2, 6])")
parser.add_argument("--size_crops", type=int, default=[32], nargs="+",
                    help="crops resolutions (example: [224, 96])")
parser.add_argument("--min_scale_crops", type=float, default=[0.2], nargs="+",
                    help="argument in RandomResizedCrop (example: [0.14, 0.05])")
parser.add_argument("--max_scale_crops", type=float, default=[1.], nargs="+",
                    help="argument in RandomResizedCrop (example: [1., 0.14])")
parser.add_argument('--batch_size', type=int, default=256, help='input batch size (default: 200)')

parser.add_argument('--temp', type=float, default=0.1)
parser.add_argument('--eps', type=float, default=0.05)
parser.add_argument('--mem_bank_n', type=int, default=16, help='No of batches in Moco membank')
parser.add_argument('--h', type=int, default=1)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--encoder_mom', type=float, default=0.9)
parser.add_argument('--loss_type', type=str, default='nce', help="Wether to run self-supervised encoder or")
parser.add_argument("--model_type", type=str, default='resnet18', help="Type of ResNet")
parser.add_argument('--batch', type=bool_flag, default=False, help='Enables automatic mixed precision')
parser.add_argument("--project_dim", type=int, default=128, help="Project embedding dimension")
parser.add_argument("--prototypes", type=int, default=1000, help="Number of prototypes")

parser.add_argument('--lr', type=float, default=[0.0005, 0.0005, 0.0005, 0.0005], help='learning rate', nargs="+")
parser.add_argument('--warmup', type=int, default=10)
parser.add_argument('--wd', type=float, default=[1e-6, 1e-6, 1e-6], nargs="+")
parser.add_argument('--amp', action='store_true', default=False, help='Enables automatic mixed precision')
parser.add_argument('--larc', action='store_true', default=False, help='Enables automatic mixed precision')
parser.add_argument('--classifiers', action='store_true', default=False,
                    help="Wether to run self-supervised encoder or classifier training task")

# parameters for output, logging, checkpointing, etc
parser.add_argument('--output_dir', type=str, default='./runs',
                    help='directory where tensorboard events and checkpoints will be stored')
parser.add_argument('--input_dir', type=str, default='/mnt/imagenet',
                    help="Input directory for the dataset. Not needed For C10,"
                         " C100 or STL10 as the data will be automatically downloaded.")
parser.add_argument('--cpt_load_path', type=str, default=None,
                    help='path from which to load checkpoint (if available)')
parser.add_argument('--cpt_name', type=str, default='mom_sim.cpt',
                    help='name to use for storing checkpoints during training')
parser.add_argument('--run_name', type=str, default='default_run',
                    help='name to use for the tensorbaord summary for this run')
parser.add_argument("--dev", type=str)
# ...
args = parser.parse_args()

if args.dev is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.dev


def main():
    # create target output dir if it doesn't exist yet
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    # torch.autograd.set_detect_anomaly(True)
    # get the dataset
    dataset = get_dataset(args.dataset)
    loss_type = get_loss_type(args.loss_type)
    encoder_size = get_encoder_size(dataset)

    # get a helper object for tensorboard logging
    log_dir = os.path.join(args.output_dir, args.run_name + '.log')
    logger = create_logger(log_dir)
    logger.info(args)
    train_loader, test_loader, num_classes = \
        build_dataset(dataset=dataset, batch_size=args.batch_size, nmb_workers=args.nmb_workers,
                      nmb_crops=args.nmb_crops, size_crops=args.size_crops, min_scale_crops=args.min_scale_crops,
                      max_scale_crops=args.max_scale_crops, path=args.data_path, a=0.5)

    torch_device = torch.device('cuda')
    checkpointer = Checkpointer(args.output_dir, args.cpt_name, logger=logger)
    if os.path.exists(os.path.join(args.output_dir, args.cpt_name)):
        cpt_load_path = os.path.join(args.output_dir, args.cpt_name)
        model = checkpointer.restore_model_from_checkpoint(cpt_load_path, training_classifier=args.classifiers)
    else:
        if args.cpt_load_path:
            model = checkpointer.restore_model_from_checkpoint(args.cpt_load_path, training_classifier=args.classifiers)
        else:
            # create new model with random parameters
            model = Model(n_classes=num_classes, encoder_size=encoder_size, prototypes=args.prototypes,
                          epoch=args.epochs, project_dim=args.project_dim, mom=args.encoder_mom, temp=args.temp,
                          eps=args.eps, loss_type=loss_type, model_type=args.model_type, batch_mlp=args.batch,
                          hidden_n=args.h, mem_bank_n=args.mem_bank_n, logger=logger)
            checkpointer.track_new_model(model)

    model = model.to(torch_device)
    # '''
    if args.classifiers:
        task = train_classifiers
    else:
        task = train_self_supervised
    task(model, args.lr, train_loader, args.nmb_crops, test_loader, logger,
         checkpointer, args.output_dir, torch_device, args.warmup, args.epochs, args.amp, args.wd, args.larc)
    # '''
    train_loader, test_loader, num_classes = \
        build_dataset(dataset=dataset, batch_size=256, nmb_workers=args.nmb_workers,
                      nmb_crops=[1], size_crops=args.size_crops[:1],
                      min_scale_crops=[1.], max_scale_crops=[1.], a=0.)

    train_classifiers(model, args.lr, train_loader, [1], test_loader, logger, checkpointer, args.output_dir,
                      torch_device, args.warmup, 200, args.amp, args.wd, args.larc)


if __name__ == "__main__":
    main()
