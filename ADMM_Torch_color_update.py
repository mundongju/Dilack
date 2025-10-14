import sys
import cv2 as cv
import time
import os
from PIL import Image
import matplotlib.pyplot as plt
import yaml
from torchvision import transforms

import torch
import numpy as np
import torch.fft as fft
import torch.nn.functional as f
import torch.nn as nn
from utils import *
from IPython.display import clear_output

import torch.nn.functional as F

from util.img_utils import clear_gray

class ADMM_LGE(nn.Module):
    def __init__(self, iterations, stacks, psf, device='cpu', infer=False, display=False, clamp_=0.1, x_0_hat=None, call_ADMM=0, Total_call=0):
        """
        Initialize the ADMM_LGE model.

        Parameters:
        - iterations (int): Number of ADMM iterations.
        - stacks (int): Number of stacks.
        - psf (numpy.ndarray or torch.Tensor): Point spread function.
        - device (str): Device to run the model on (default is 'cpu').
        - infer (bool): If True, initialize with fixed values for mu1, mu2, mu3, tau.
        - display (bool): If True, display intermediate results during ADMM iterations.
        - clamp_ (float): Clamping parameter.
        """
        super(ADMM_LGE, self).__init__()
        self.device = device
        self.display = display
        self.clamp_ = clamp_
        self.x_0_hat = x_0_hat

        # Trainable parameters
        if infer:
            #self.mu1 = [1e-1]
            #self.mu2 = [1e-1]
            #self.mu3 = [1e-1]
            #self.tau = [1e-1]
            
            self.mu1 = [1e-7]
            self.mu2 = [1e-7]
            self.mu3 = [1e-7]
            if call_ADMM == 0:
                self.tau = [1e-7]
            else:
                self.tau = [1e-7 * (1-call_ADMM/Total_call)]
                print("tau!!!!" , self.tau[0])


        else:
            self.mu1 = torch.nn.Parameter(torch.tensor(np.ones(stacks) * 6.754827e-09, dtype=torch.float32))
            self.mu2 = torch.nn.Parameter(torch.tensor(np.ones(stacks) * 1.164068e-08, dtype=torch.float32))
            self.mu3 = torch.nn.Parameter(torch.tensor(np.ones(stacks) * 1.5371798e-08, dtype=torch.float32))
            self.tau = torch.nn.Parameter(torch.tensor(np.ones(stacks) * 5.5435e-07, dtype=torch.float32))

        # Load psf
        if not torch.is_tensor(psf):
            psf = to_tensor_or_numpy(psf)
        self.psf = psf.to(self.device)
        self.psf = self.psf / torch.linalg.norm(self.psf.contiguous().view(-1))
        self.sensor_size = psf.shape
        self.full_size = tuple(x for x in psf.shape[:-2]) + tuple(2*x for x in psf.shape[-2:])
        self.iter = iterations

    def ADMM(self, x_in):
        """
        Perform ADMM iterations on the input.

        Parameters:
        - x_in (numpy.ndarray or torch.Tensor): Input data.

        Returns:
        - torch.Tensor: Processed output.
        """
        self.raw = x_in
        if not torch.is_tensor(self.raw):
            self.raw = to_tensor_or_numpy(self.raw)
        self.raw = self.raw.to(self.device)
        self.raw = self.raw / torch.linalg.norm(self.raw.ravel())

        X, U, self.V, W, xi, eta, rho = self.init_Matrices()
        total_time = 0

        for i in range(self.iter):
            self.i = 0
            start_time = time.time()
            X, U, self.V, W, xi, eta, rho = self.ADMM_Step(X, U, self.V, W, xi, eta, rho)

            img = self.Crop(torch.clone(self.V), self.full_size, self.sensor_size)
            iteration_time = time.time() - start_time
            total_time = total_time + iteration_time



            if self.display:
                clear_output(wait=True)
                plot_ADMM(
                    self.psf,
                    self.raw,
                    img,
                    times=iteration_time,
                    iteration=i,
                    total_time=total_time,
                    clamp_=self.clamp_,
                    hyperparameters={'mu1': self.mu1, 'mu2': self.mu2, 'mu3': self.mu3, 'tau': self.tau}
                )

        return img

    def init_Matrices(self):
        """
        Initialize matrices and variables for ADMM.

        Returns:
        - Tuple of torch.Tensor: Initialized matrices and variables.
        """
        self.H_fft = self.fft_shift_2d(self.CropT(self.psf, self.full_size, self.sensor_size))
        X_divmat_num = self.CropT(torch.ones(self.sensor_size).to(self.device), self.full_size, self.sensor_size)
        self.X_divmat = torch.ones_like(X_divmat_num).to(self.device) / (X_divmat_num + self.mu1[0]) # CtC+mu1I

        MTM_component = self.mu1[0] * (torch.abs(torch.conj(self.H_fft) * self.H_fft))

        PsiTPsi = torch.zeros(self.full_size).to(self.device)
        PsiTPsi[...,0,0] = 4
        PsiTPsi[...,0,1] = PsiTPsi[...,1,0] = PsiTPsi[...,0,-1] = PsiTPsi[...,-1,0] = -1
        PsiTPsi = fft.fftn(PsiTPsi, dim=(-2,-1))

        self.R_divmat = torch.ones_like(MTM_component).to(self.device) / (
                MTM_component + self.mu2[0] * torch.abs(PsiTPsi) + self.mu3[0])

        # 09/25 modify init_matrices
        # breakpoint()
        x0_gray = 0.2989 * self.x_0_hat[:, 0, :, :] + 0.5870 * self.x_0_hat[:, 1, :, :] + 0.1140 * self.x_0_hat[:, 2, :, :]
        x0_gray =  (self.x_0_hat[:, 0, :, :]+ self.x_0_hat[:, 1, :, :] + self.x_0_hat[:, 2, :, :])/3
        x0_gray = x0_gray.unsqueeze(0)  # 차원을 [1, 1, 256, 256]으로 변경
        pad_height = (900 - 256) // 2
        pad_width = (900 - 256) // 2
        x0_resized = torch.nn.functional.interpolate(x0_gray, size=(900, 900), mode='bilinear', align_corners=False)
        x0_resized_norm = (1+x0_resized)/2
        # plt.imsave("./x0_resized.png", clear_gray(x0_resized_norm))
        # self.V = x0_resized.squeeze(1).to(self.device)  # 채널 차원을 제거하여 [1, 900, 900]으로 맞춤

        #test
        self.V = (x0_resized.squeeze(1).to(self.device)) * 0.001
        # self.V = torch.zeros(self.full_size).to(self.device) + 0.01
        
        # # U 초기화
        # U = self.Psi(self.V)
    
        # # X 초기화
        # X = self.V.clone()
        # X = x0_resized.squeeze(1).to(self.device)  # 채널 차원을 제거하여 [1, 900, 900]으로 맞춤
        # X = 0.5 * torch.ones(self.full_size).to(self.device)
    
        # # W 초기화
        # W = self.V.clone()

        # false rev]
        # H_fft_resized = self.H_fft.unsqueeze(0)  # 결과: [1, 1, 900, 900]
        # H_fft_resized = F.interpolate(H_fft_resized, size=(256, 256), mode='bilinear', align_corners=False)  # 결과: [1, 1, 256, 256]
        # X = self.X_update(torch.zeros_like(self.M(self.V, self.H_fft)).to(self.device), self.V, self.raw)
        # U = self.U_update(torch.zeros_like(self.Psi(self.V)).to(self.device), self.V)
        # W = torch.maximum(torch.zeros_like(self.V).to(self.device), self.V)

        # origin]
        # self.V = torch.zeros(self.full_size).to(self.device)
        
        X = torch.zeros(self.full_size).to(self.device)
        U = self.Psi(self.V)
        W = self.V
        # U = torch.zeros(self.full_size + (2,)).to(self.device)
        # W = torch.zeros(self.full_size).to(self.device)
        xi = torch.zeros_like(self.M(self.V, self.H_fft)).to(self.device)
        eta = torch.zeros_like(self.Psi(self.V)).to(self.device)
        rho = torch.zeros_like(W).to(self.device)
        # xi = self.mu1[0] * (self.M(self.V, self.H_fft) - X)
        # eta = self.mu2[0] * (self.Psi(self.V) - U)
        # rho = self.mu3[0] * (self.V - W)

        return X, U, self.V, W, xi, eta, rho
    
    def ADMM_Step(self, X, U, V, W, xi, eta, rho):
        #U = self.isotropic_tv_update(U, eta, self.tau[self.i])
        with torch.no_grad():
            U = self.U_update(eta, self.V)
            X = self.X_update(xi, V, self.raw)
            self.V = self.V_update(W, rho, U, eta, X, xi)
            W = self.W_update(rho, self.V)
            xi = self.xi_update(xi, self.V, X)
            eta = self.eta_update(eta, self.V, U)
            rho = self.rho_update(rho, self.V, W)
        return X, U, self.V, W, xi, eta, rho

    def U_update(self, eta, image_est):
        return self.SoftThresh(self.Psi(image_est) + eta / self.mu2[self.i], self.tau[self.i] / self.mu2[self.i])

    def X_update(self, xi, image_est, sensor_reading):
        return self.X_divmat * (xi + self.mu1[self.i] * self.M(image_est, self.H_fft) +
                                self.CropT(sensor_reading, self.full_size, self.sensor_size))

    def W_update(self, rho, image_est):
        return torch.maximum(rho / self.mu3[self.i] + image_est, torch.zeros_like(image_est).to(self.device))

    def V_update(self, w, rho, u, eta, x, xi):
        freq_space_result = self.R_divmat * self.fft_shift_2d(self.r_calc(w, rho, u, eta, x, xi))
        return torch.real(self.ifft_shift_2d(freq_space_result))

    def r_calc(self, w, rho, u, eta, x, xi):
        return (self.mu3[self.i] * w - rho) + self.PsiT(self.mu2[self.i] * u - eta) + self.MT(
            self.mu1[self.i] * x - xi, self.H_fft)

    def xi_update(self, xi, V, X):
        return xi + self.mu1[self.i] * (self.M(V, self.H_fft) - X)

    def eta_update(self, eta, V, U):
        return eta + self.mu2[self.i] * (self.Psi(V) - U)

    def rho_update(self, rho, V, W):
        return rho + self.mu3[self.i] * (V - W)

    def SoftThresh(self, x, tau_c):
        theta = tau_c * 0.1
        return torch.sign(x) * (
                (torch.abs(x) - tau_c - theta) + torch.sqrt(
            torch.pow((torch.abs(x) - tau_c - theta), 2) + 4 * theta * torch.abs(x))) / 2

    def Psi(self, v):
        return torch.stack((torch.roll(v, 1, dims = -2) - v, torch.roll(v, 1, dims = -1) - v), dim = -1)

    def PsiT(self, U):
        diff1 = torch.roll(U[..., 0], -1, dims=-2) - U[..., 0]
        diff2 = torch.roll(U[..., 1], -1, dims=-1) - U[..., 1]
        return diff1 + diff2

    def fft_shift_2d(self, *args, **kwargs):        
        return fft.fftn(fft.ifftshift(*args, dim=(-2,-1)), dim=(-2,-1))

    def ifft_shift_2d(self, *args, **kwargs):
        return fft.fftshift(fft.ifftn(*args, dim=(-2,-1)), dim=(-2,-1))

    def M(self, vk, H_fft):
        return torch.real(self.ifft_shift_2d((self.fft_shift_2d(vk) * H_fft)))

    def MT(self, x, H_fft):
        return torch.real(self.ifft_shift_2d((self.fft_shift_2d(x) * torch.conj(H_fft))))

    def Crop(self, M, full_size, sensor_size):
        top = (full_size[-2] - sensor_size[-2]) // 2
        bottom = (full_size[-2] + sensor_size[-2]) // 2
        left = (full_size[-1] - sensor_size[-1]) // 2
        right = (full_size[-1] + sensor_size[-1]) // 2
        return M[...,top:bottom,left:right]

    def CropT(self, b, full_size, sensor_size):
        v_pad = (full_size[-2] - sensor_size[-2]) // 2
        h_pad = (full_size[-1] - sensor_size[-1]) // 2
        return f.pad(b, (h_pad, h_pad, v_pad, v_pad))

    def forward(self, x):
        out = self.ADMM(x)
        return out

    def isotropic_tv_norm(self, v):
        # 이미지 v의 각 픽셀에서 변화량을 계산합니다.
        v_grad = self.Psi(v)
        # L2 노름을 계산합니다.
        tv_norm = torch.sqrt(torch.sum(v_grad ** 2, dim=-1) + 1e-8)
        return tv_norm

    def isotropic_tv_update(self, U, eta, tau):
        # U를 이용해 그라디언트를 계산하고, eta를 더해준 다음 tau를 사용해 정규화 합니다.
        U_grad = self.Psi(U)
        grad_mag = torch.sqrt(torch.sum(U_grad ** 2, dim=-1) + 1e-8)
        reducer = torch.max(torch.tensor(0.0, device=self.device), 1 - tau / (grad_mag + 1e-8))
        U_grad_updated = U_grad * reducer.unsqueeze(-1)
        # 정규화된 그라디언트를 이용해 U를 업데이트합니다.
        return U - self.PsiT(U_grad_updated)
