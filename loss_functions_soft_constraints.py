import torch
import torch.nn.functional as F

import diff_operators
import modules, utils

import math
import numpy as np


def initialize_hji_air3D_exact_soft(dataset, minWith):
    # Initialize the loss function for the air3D problem
    # The dynamics parameters
    velocity = dataset.velocity
    omega_max = dataset.omega_max
    alpha_angle = dataset.alpha_angle

    # The derivation of the loss function is as follows:
    # V(x,t) = l(x) + NN(x,t)
    # For 2D case, Hamiltonian = p1(−ve + vpcosx3) + p2(vpsinx3) + w||p1x2 − p2x1 − p3|| + wp3, where p1 and p2 are derivatives of V wrt x and y respectively
    # p1 = lx_grad(x) + d(NN)/dx
    # p2 = lx_grad(y) + d(NN)/dy
    # p3 = d(NN)/d(theta)
    # PDE Loss = dV/dt + ham which is equal to:
    # PDE Loss = d(NN)/dt + ham (we add the hamiltonian because we are computing the goal set)
    # HJI VI: min(PDE Loss, V(x,t) - l(x)) which is equal to:
    # HJI VI: min(PDE Loss, NN(x,t))
    # Boundary Loss: V(x,0) - l(x) = NN(x,0)

    def hji_air3D_exact_soft(model_output, gt):
        x = model_output['model_in']  # (meta_batch_size, num_points, 4)
        y = model_output['model_out']  # (meta_batch_size, num_points, 1)
        dirichlet_mask = gt['dirichlet_mask']
        lx_grads = gt['lx_grads']
        batch_size = x.shape[1]
        #print(lx_grads.shape)
        #print(lx_grads[..., 0])
        #print(lx_grads[..., 1])

        du, status = diff_operators.jacobian(y, x)
        dudt = du[..., 0, 0]
        dudx = du[..., 0, 1:]
        #print(dudx.shape)

        x_theta = x[..., 3] * 1.0

        # Scale the costate for theta appropriately to align with the range of [-pi, pi]
        dudx[..., 2] = dudx[..., 2] / alpha_angle
        # Scale the coordinates
        x_theta = alpha_angle * x_theta

        # Air3D dynamics
        # \dot x    = -v_a + v_b \cos \psi + a y
        # \dot y    = v_b \sin \psi - a x
        # \dot \psi = b - a
        der_x = lx_grads[..., 0] + dudx[..., 0]
        der_y = lx_grads[..., 1] + dudx[..., 1]
        der_theta = dudx[..., 2]

        # Compute the hamiltonian for the ego vehicle
        ham = omega_max * torch.abs(der_x * x[..., 2] - der_y * x[..., 1] - der_theta)  # Control component
        ham = ham - omega_max * torch.abs(der_theta)  # Disturbance component
        ham = ham + (velocity * (torch.cos(x_theta) - 1.0) * der_x) + (velocity * torch.sin(x_theta) * der_y)  # Constant component

        # If we are computing BRT then take min with zero
        if minWith == 'zero':
            ham = torch.clamp(ham, max=0.0)

        if torch.all(dirichlet_mask):
            diff_constraint_hom = torch.Tensor([0])
        else:
            diff_constraint_hom = dudt - ham
            if minWith == 'target':
                diff_constraint_hom = torch.max(diff_constraint_hom[:, :, None], y)

        dirichlet = y[dirichlet_mask]

        # A factor of 15e2 to make loss roughly equal
        return {'dirichlet': torch.abs(dirichlet).sum() * batch_size / 15e2,
                'diff_constraint_hom': torch.abs(diff_constraint_hom).sum()}

    return hji_air3D_exact_soft


def initialize_hji_2D_example_exact_soft(dataset, minWith):
    # Initialize the loss function for the air3D problem
    # The dynamics parameters
    velocity = dataset.velocity

    # The derivation of the loss function is as follows:
    # V(x,t) = l(x) + NN(x,t)
    # For 2D case, Hamiltonian = sqrt(p1**2 + p2**2) where p1 and p2 are derivatives of V wrt x and y respectively
    # p1 = lx_grad(x) + d(NN)/dx
    # p2 = lx_grad(y) + d(NN)/dy
    # PDE Loss = dV/dt + ham which is equal to:
    # PDE Loss = d(NN)/dt + ham (we add the hamiltonian because we are computing the goal set)
    # HJI VI: min(PDE Loss, V(x,t) - l(x)) which is equal to:
    # HJI VI: min(PDE Loss, NN(x,t))
    # Boundary Loss: V(x,0) - l(x) = NN(x,0)

    def hji_2D_example_exact_soft(model_output, gt):
        x = model_output['model_in']  # (meta_batch_size, num_points, 3)
        y = model_output['model_out']  # (meta_batch_size, num_points, 1)
        dirichlet_mask = gt['dirichlet_mask']
        lx_grads = gt['lx_grads']
        batch_size = x.shape[1]

        du, status = diff_operators.jacobian(y, x)
        dudt = du[..., 0, 0]
        dudx = du[..., 0, 1:]

        der_x = lx_grads[..., 0] + dudx[..., 0]
        der_y = lx_grads[..., 1] + dudx[..., 1]

        der_x_square = (der_x) * (der_x)
        der_y_square = (der_y) * (der_y)
        value_der = torch.sqrt(der_x_square + der_y_square)

        ham = value_der * velocity

        # If we are computing BRT then take min with zero
        if minWith == 'zero':
            ham = torch.clamp(ham, max=0.0)

        if torch.all(dirichlet_mask):
            diff_constraint_hom = torch.Tensor([0])
        else:
            diff_constraint_hom = dudt + ham
            if minWith == 'target':
                diff_constraint_hom = torch.max(diff_constraint_hom[:, :, None], y)

        dirichlet = y[dirichlet_mask]

        # A factor of 15e2 to make loss roughly equal
        return {'dirichlet': torch.abs(dirichlet).sum() * batch_size / 15e2,
                'diff_constraint_hom': torch.abs(diff_constraint_hom).sum()}

    return hji_2D_example_exact_soft
