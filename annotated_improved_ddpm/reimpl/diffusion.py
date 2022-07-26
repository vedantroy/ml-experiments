import torch as th
import torch.nn.functional as F
import math

# Simple-ish Gaussian Diffusion
# Notes:
# - Learns the variance
#  - Scales the variance loss to avoid L_vlb overwhelming L_simple
#  - Applies a stop-gradient to the mean term for L_vlb to only allow
#    backpropagation through the variance term
# - Estimates the noise (epsilon)
# - Uses a hybrid objective (L_simple + L_vlb) without resampling


def cosine_betas(timesteps, s=0.008, max_beta=0.999):
    """
    Get B_t for the cosine schedule (eq 17)

    :param max_beta: "In practice, we clip B_t to be no larger than 0.999 to prevent
                      singularities at the end of the diffusion process"
    :param s: "We use a small offset s to prevent B_t from being too small near t = 0"
    """
    # If we add noise twice, then there are 3 total states (0, 1, 2)
    states = timesteps + 1
    t = th.linspace(start=0, end=timesteps, steps=states, dtype=th.float64)
    f_t = th.cos(((t / timesteps) + s) / (1 + s) * (math.pi / 2)) ** 2
    alphas_cumprod = f_t / f_t[0]
    alphas_cumprod_t = alphas_cumprod[1:]
    alphas_cumprod_t_minus_1 = alphas_cumprod[:-1]
    return 1 - (alphas_cumprod_t / alphas_cumprod_t_minus_1)


def extract_for_timesteps(x, timesteps, broadcast_shape):
    # This will most certainly need to be fixed later
    vals = x[timesteps]
    return vals

def make_q_posterior_mean_variance(self, alphas_cumprod, alphas_cumprod_prev, betas):
    # (10) in [0]
    posterior_variance = (
        (1 - self.alphas_cumprod_prev) / (1 - alphas_cumprod)
    ) * betas

    # (11) in [0]
    posterior_mean_coef_x_start = (alphas_cumprod_prev.sqrt() * betas) / (
        1 - alphas_cumprod
    )
    posterior_mean_coef_x_t = (
        (alphas_cumprod.sqrt() * (1 - alphas_cumprod_prev)) / (1 - alphas_cumprod)
    )

    # (12) in [0]
    def q_posterior_mean_variance(self, x_t, x_start, t):
        """
        Calculate the mean and variance of the normal distribution q(x_{t-1}|x_t, x_0)

        Use this to get the previous step in the diffusion process, given the current step
        (12) in [0]

        :param x_t: The result of the next step in the diffusion process (batch)
        :param x_start: The initial image (batch)
        :param t: The current timestep (batch)
        """
        mean = (
            extract_for_timesteps(self.posterior_mean_coef_x_start, t) * x_start
            + extract_for_timesteps(self.posterior_mean_coef_x_t, t) * x_t
        )
        return mean, extract_for_timesteps(self.posterior_variance, t)
    return q_posterior_mean_variance

class GaussianDiffusion:
    def __init__(self, *, betas):
        # For the cosine schedule, we end up recalculating the alphas
        # but it's more elegant to only be provided the betas
        alphas = 1 - betas
        alphas_cumprod = th.cumprod(alphas)
        sqrt_alphas_cumprod = th.sqrt(alphas_cumprod)

        # TODO(verify): By prepending 1, the 1st beta is 0
        # This represents the initial image, which as a mean but no variance (since it's ground truth)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        sqrt_alphas_cumprod_prev = th.sqrt(alphas_cumprod_prev)

        # (10) in [0]
        self.posterior_variance = (
            (1 - self.alphas_cumprod_prev) / (1 - alphas_cumprod)
        ) * betas
        self.posterior_mean_coef_x_start = (sqrt_alphas_cumprod_prev * betas) / (
            1 - alphas_cumprod
        )
        self.posterior_mean_coef_x_t = (
            (sqrt_alphas_cumprod * (1 - alphas_cumprod_prev)) / (1 - alphas_cumprod)
        )

    def q_posterior_mean_variance(self, x_t, x_start, t):
        """
        Calculate the mean and variance of the normal distribution q(x_{t-1}|x_t, x_0)

        Use this to get the previous step in the diffusion process, given the current step
        (12) in [0]

        :param x_t: The result of the next step in the diffusion process (batch)
        :param x_start: The initial image (batch)
        :param t: The current timestep (batch)
        """
        mean = (
            extract_for_timesteps(self.posterior_mean_coef_x_start, t) * x_start
            + extract_for_timesteps(self.posterior_mean_coef_x_t, t) * x_t
        )
        return mean, extract_for_timesteps(self.posterior_variance, t)

    def vlb_loss_terms(self):
        pass

    def q_sample(self, x_start, t, noise):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0)

        :param x_start: The initial image (batch)
        :param t: The timestep to diffuse to (batch)
        :param noise: The noise (epsilon in the paper)
        """

        N = x_start.shape[0]
        assert t.shape == (N,)

        # (9) in [0]
        mean = extract_for_timesteps(self.sqrt_alphas_cumprod[t], t, None) * x_start
        var = extract_for_timesteps(self.sqrt_alphas_cumprod_minus_1[t], t, None)
        return mean + var * noise

    def training_loss(self, x_start, t):
        noise = th.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise)

        # Get the model's noise prediction conditioned on input and t
        # Split the model output into 2 components:
        #  - the noise estimate
        #  - the variance estimate
        # Both of these have dimensions B x C x H x W
        # Do MSE between model's noise & actual noise
        # Calculate the VLB loss & scale it (avoids overwhelming the MSE loss)
        #   The VLB loss will be calculated from all of the model's outputs (mean + variance)
        #   But we apply a stop-gradient to the mean term to only allow backpropagation through the variance term
        # Apparently?? we can calculate the initial image from the noise estimate
