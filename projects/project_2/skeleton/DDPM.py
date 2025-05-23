import torch
import random
import numpy as np
import torch.nn as nn
from ResUNet import ConditionalUnet


device = torch.device(
    "mps"
    if torch.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
print(f"Using device: {device}")


class ConditionalDDPM(nn.Module):
    def __init__(self, dmconfig):
        super().__init__()
        self.dmconfig = dmconfig
        self.loss_fn = nn.MSELoss()
        self.network = ConditionalUnet(
            1, self.dmconfig.num_feat, self.dmconfig.num_classes
        ).to(device)
        (
            self.betas,
            self.alphas,
            self.sqrt_betas,
            self.oneover_sqrt_alphas,
            self.alpha_bars,
        ) = self.precompute_scheduler()

    def precompute_scheduler(self):
        beta_1, beta_T, T = self.dmconfig.beta_1, self.dmconfig.beta_T, self.dmconfig.T
        betas = torch.tensor(
            np.linspace(beta_1, beta_T, T), dtype=torch.float32, device=device
        )

        alphas = 1 - betas
        sqrt_betas = torch.sqrt(betas)
        oneover_sqrt_alphas = 1 / torch.sqrt(alphas)
        alpha_bars = torch.cumprod(alphas, dim=0)

        return (betas, alphas, sqrt_betas, oneover_sqrt_alphas, alpha_bars)

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
        sqrt_alpha_bar = torch.sqrt(alpha_t_bar)
        sqrt_oneminus_alpha_bar = torch.sqrt(1 - alpha_t_bar)

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

        self.network.train()
        B = conditions.shape[0]

        # convert conditions to one-hot vector
        conditions = torch.nn.functional.one_hot(
            conditions, num_classes=self.dmconfig.num_classes
        ).to(device)

        # set (mask_p * B) conditions to unconditional values
        n_uncond = int(self.dmconfig.mask_p * B)
        uncond_indices = torch.tensor(random.sample(range(B), n_uncond), device=device)

        conditions[uncond_indices] = self.dmconfig.condition_mask_value

        # sample times
        times = torch.randint(low=1, high=T + 1, size=(B, 1, 1, 1), device=device)

        # sample gaussian noise
        epsilon = torch.randn(size=images.shape, device=device)

        # corrupt images to timesteps
        schedule = self.scheduler(times)
        sqrt_alpha_bar = schedule["sqrt_alpha_bar"]
        sqrt_oneminus_alpha_bar = schedule["sqrt_oneminus_alpha_bar"]

        X_t = sqrt_alpha_bar * images + sqrt_oneminus_alpha_bar * epsilon

        # normalize times
        times = times.type(torch.float) / T

        # compute
        out = self.network(X_t, times, conditions)

        noise_loss = self.loss_fn(out, epsilon)

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

        # standard normal sample
        self.network.eval()
        B = conditions.shape[0]
        X_t = torch.randn(
            size=(B, self.dmconfig.num_channels, *self.dmconfig.input_dim),
            device=device,
        )

        # convert conditions to one-hot
        conditions = torch.nn.functional.one_hot(
            conditions, self.dmconfig.num_classes
        ).to(device)

        # prepare unconditional input
        unconditional_conditions = (
            torch.ones_like(conditions) * self.dmconfig.condition_mask_value
        )

        with torch.no_grad():
            for t in torch.arange(T, 0, -1, device=device):
                if t > 1:
                    z = torch.randn(X_t.shape, device=device)
                else:
                    z = 0

                # normalize t
                t_norm = torch.ones((B, 1, 1, 1), device=device) * t
                t_norm = t_norm / T

                # get noise
                conditional_noise = self.network(X_t, t_norm, conditions)
                unconditional_noise = self.network(
                    X_t, t_norm, unconditional_conditions
                )
                noise_t = (1 + omega) * conditional_noise - omega * unconditional_noise

                # generate next image
                oneover_sqrt_alpha = self.oneover_sqrt_alphas[t - 1]
                oneminus_alpha = 1 - self.alphas[t - 1]
                oneover_sqrt_oneminus_alpha_bar = 1 / torch.sqrt(
                    1 - self.alpha_bars[t - 1]
                )
                sqrt_beta = self.sqrt_betas[t - 1]
                X_t = (
                    oneover_sqrt_alpha
                    * (X_t - oneminus_alpha * oneover_sqrt_oneminus_alpha_bar * noise_t)
                    + sqrt_beta * z
                )

        # ==================================================== #
        generated_images = (X_t * 0.3081 + 0.1307).clamp(
            0, 1
        )  # denormalize the output images
        return generated_images
