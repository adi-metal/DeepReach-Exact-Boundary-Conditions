# Enable import from parent package
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
import os

#sys.path.append('../.')

import dataio, utils, training, loss_functions_soft_constraints, modules

import torch
import numpy as np
import math
from torch.utils.data import DataLoader
import configargparse


p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--logging_root', type=str, default='./logs',
               help='root for logging')
p.add_argument('--experiment_name', type=str, required=True,
               default='./summaries',
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

# General training options
p.add_argument('--batch_size', type=int, default=32)
p.add_argument('--lr', type=float, default=2e-5, help='learning rate. default=2e-5')
p.add_argument('--num_epochs', type=int, default=100000,
               help='Number of epochs to train for.')

p.add_argument('--epochs_til_ckpt', type=int, default=1000,
               help='Time interval in seconds until checkpoint is saved.')
p.add_argument('--steps_til_summary', type=int, default=100,
               help='Time interval in seconds until tensorboard summary is saved.')
p.add_argument('--model', type=str, default='sine', required=False, choices=['sine', 'tanh', 'sigmoid', 'relu'],
               help='Type of model to evaluate, default is sine.')
p.add_argument('--mode', type=str, default='mlp', required=False, choices=['mlp', 'rbf', 'pinn'],
               help='Whether to use uniform velocity parameter')
p.add_argument('--tMin', type=float, default=0.0, required=False, help='Start time of the simulation')
p.add_argument('--tMax', type=float, default=0.5, required=False, help='End time of the simulation')
p.add_argument('--num_hl', type=int, default=3, required=False, help='The number of hidden layers')
p.add_argument('--num_nl', type=int, default=512, required=False, help='Number of neurons per hidden layer.')
p.add_argument('--pretrain_iters', type=int, default=2000, required=False, help='Number of pretrain iterations')
p.add_argument('--counter_start', type=int, default=-1, required=False,
               help='Defines the initial time for the curriculul training')
p.add_argument('--counter_end', type=int, default=-1, required=False,
               help='Defines the linear step for curriculum training starting from the initial time')
p.add_argument('--num_src_samples', type=int, default=1000, required=False,
               help='Number of source samples at each time step')

p.add_argument('--velocity', type=float, default=0.6, required=False, help='Speed of the dubins car')
p.add_argument('--collisionR', type=float, default=0.75, required=False, help='Collision radius between vehicles')
p.add_argument('--minWith', type=str, default='none', required=False, choices=['none', 'zero', 'target'],
               help='BRS vs BRT computation')

p.add_argument('--clip_grad', default=0.0, type=float, help='Clip gradient.')
p.add_argument('--use_lbfgs', default=False, type=bool, help='use L-BFGS.')
p.add_argument('--pretrain', action='store_true', default=False, required=False, help='Pretrain dirichlet conditions')

p.add_argument('--seed', type=int, default=0, required=False, help='Seed for the simulation.')

p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')
p.add_argument('--checkpoint_toload', type=int, default=0, help='Checkpoint from which to restart the training.')
opt = p.parse_args()

# Set the source coordinates for the target set and the obstacle sets
source_coords = [0., 0., 0.]
if opt.counter_start == -1:
    opt.counter_start = opt.checkpoint_toload

if opt.counter_end == -1:
    opt.counter_end = opt.num_epochs

dataset = dataio.Reachability2DSource(numpoints=65000, collisionR=opt.collisionR, velocity=opt.velocity,
                                         pretrain=opt.pretrain, tMin=opt.tMin,
                                         tMax=opt.tMax, counter_start=opt.counter_start, counter_end=opt.counter_end,
                                         pretrain_iters=opt.pretrain_iters, seed=opt.seed,
                                         num_src_samples=opt.num_src_samples)

dataloader = DataLoader(dataset, shuffle=True, batch_size=opt.batch_size, pin_memory=True, num_workers=0)

model = modules.SingleBVPNet(in_features=3, out_features=1, type=opt.model, mode=opt.mode,
                             final_layer_factor=1., hidden_features=opt.num_nl, num_hidden_layers=opt.num_hl)
model.cuda()

# Define the loss
loss_fn = loss_functions_soft_constraints.initialize_hji_2D_example_exact_soft(dataset, opt.minWith)

root_path = os.path.join(opt.logging_root, opt.experiment_name)


def val_fn(model, ckpt_dir, epoch):
    # Time values at which the function needs to be plotted
    times = [0., 0.5 * (opt.tMax - 0.1), (opt.tMax - 0.1)]
    num_times = len(times)

    # Create a figure
    fig = plt.figure(figsize=(5 * num_times, 5 * num_times))

    # Get the meshgrid in the (x, y) coordinate
    sidelen = 200
    mgrid_coords = dataio.get_mgrid(sidelen)
    signed_distance = torch.norm(mgrid_coords, dim=1, keepdim=True) - opt.collisionR 
    signed_distance = signed_distance.reshape(sidelen, sidelen)
    signed_distance = signed_distance.detach().cpu().numpy()

    # Start plotting the results
    for i in range(num_times):
        time_coords = torch.ones(mgrid_coords.shape[0], 1) * times[i]

        coords = torch.cat((time_coords, mgrid_coords), dim=1)
        model_in = {'coords': coords.cuda()}
        model_out = model(model_in)['model_out']

        # Detatch model ouput and reshape
        model_out = model_out.detach().cpu().numpy()
        model_out = model_out.reshape((sidelen, sidelen))
        model_out = model_out + signed_distance

        # Unnormalize the value function
        norm_to = 0.02
        mean = 0.25
        var = 0.5
        model_out = (model_out * var / norm_to) + mean

        # Plot the zero level sets
        model_out = (model_out <= 0.001) * 1.

        # Plot the actual data
        ax = fig.add_subplot(num_times, 1, i+1)
        ax.set_title('t = %0.2f' % (times[i]))
        s = ax.imshow(model_out.T, cmap='bwr', origin='lower', extent=(-1., 1., -1., 1.))
        fig.colorbar(s)

    fig.savefig(os.path.join(ckpt_dir, 'BRS_validation_plot_epoch_%04d.png' % epoch))


training.train(model=model, train_dataloader=dataloader, epochs=opt.num_epochs, lr=opt.lr,
               steps_til_summary=opt.steps_til_summary, epochs_til_checkpoint=opt.epochs_til_ckpt,
               model_dir=root_path, loss_fn=loss_fn, clip_grad=opt.clip_grad,
               use_lbfgs=opt.use_lbfgs, validation_fn=val_fn, start_epoch=opt.checkpoint_toload)
