import torch
import numpy as np
import torch.nn as nn
from ResUNet import ConditionalUnet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ConditionalDDPM(nn.Module):
    def __init__(self, dmconfig):
        super().__init__()
        self.dmconfig = dmconfig
        self.loss_fn = nn.MSELoss()
        self.network = ConditionalUnet(
            1, self.dmconfig.num_feat, self.dmconfig.num_classes
        )
        (
            self.betas,
            self.alphas,
            self.sqrt_betas,
            self.oneover_sqrt_alphas,
            self.alpha_bars,
        ) = self.precompute_scheduler()

    def precompute_scheduler(self):
        beta_1, beta_T, T = self.dmconfig.beta_1, self.dmconfig.beta_T, self.dmconfig.T
        betas = np.linspace(beta_1, beta_T, T)
        alphas = 1 - betas
        sqrt_betas = np.sqrt(betas)
        oneover_sqrt_alphas = 1 / np.sqrt(alphas)
        alpha_bars = torch.cumprod(torch.tensor(alphas), dim=0)
        return betas, alphas, sqrt_betas, oneover_sqrt_alphas, alpha_bars

    def scheduler(self, t_s):
        beta_1, beta_T, T = self.dmconfig.beta_1, self.dmconfig.beta_T, self.dmconfig.T
        # ==================================================== #
        # YOUR CODE HERE:
        #   Inputs:
        #       t_s: the input time steps, with shape (B,1).
        #   Outputs:
        #       one dictionary containing the variance schedule
        #       $\beta_t$ along with other potentially useful constants.

        idx = t_s - 1  # for indexing
        beta_t = self.betas[idx]
        sqrt_beta_t = self.sqrt_betas[idx]
        alpha_t = self.alphas[idx]
        oneover_sqrt_alpha = self.oneover_sqrt_alphas[idx]
        alpha_t_bar = self.alpha_bars[idx]
        sqrt_alpha_bar = np.sqrt(alpha_t_bar)
        sqrt_oneminus_alpha_bar = np.sqrt(1 - alpha_t_bar)

        # ==================================================== #
        return {
            "beta_t": beta_t,
            "sqrt_beta_t": sqrt_beta_t,
            "alpha_t": alpha_t,
            "sqrt_alpha_bar": sqrt_alpha_bar,
            "oneover_sqrt_alpha": oneover_sqrt_alpha,
            "alpha_t_bar": alpha_t_bar,
            "sqrt_oneminus_alpha_bar": sqrt_oneminus_alpha_bar,
        }

    def forward(self, images, conditions):
        T = self.dmconfig.T
        noise_loss = None
        # ==================================================== #
        # YOUR CODE HERE:
        #   Complete the training forward process based on the
        #   given training algorithm.
        #   Inputs:
        #       images: real images from the dataset, with size (B,1,28,28).
        #       conditions: condition labels, with size (B). You should
        #                   convert it to one-hot encoded labels with size (B,10)
        #                   before making it as the input of the denoising network.
        #   Outputs:
        #       noise_loss: loss computed by the self.loss_fn function  .

        conditions = torch.nn.functional.one_hot(conditions, self.dmconfig.num_classes)
        times = torch.randint(low=1, high=T + 1, size=(images.shape[0], 1, 1, 1))
        out = self.network(images, times, conditions)

        target = torch.randn(images.shape)
        noise_loss = self.loss_fn(out, target)
        # ==================================================== #

        return noise_loss

    def sample(self, conditions, omega):
        T = self.dmconfig.T
        X_t = None
        # ==================================================== #
        # YOUR CODE HERE:
        #   Complete the training forward process based on the
        #   given sampling algorithm.
        #   Inputs:
        #       conditions: condition labels, with size (B). You should
        #                   convert it to one-hot encoded labels with size (B,10)
        #                   before making it as the input of the denoising network.
        #       omega: conditional guidance weight.
        #   Outputs:
        #       generated_images

        pass

        # ==================================================== #
        generated_images = (X_t * 0.3081 + 0.1307).clamp(
            0, 1
        )  # denormalize the output images
        return generated_images

