import types

import torch as th
import torch.nn.functional as F
from einops import rearrange
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


def make_q_posterior_mean_variance(
    alphas, alphas_cumprod, alphas_cumprod_prev, betas, posterior_variance
):
    # (11) in [0]
    posterior_mean_coef_x_0 = (alphas_cumprod_prev.sqrt() * betas) / (
        1 - alphas_cumprod
    )
    posterior_mean_coef_x_t = (alphas.sqrt() * (1 - alphas_cumprod_prev)) / (
        1 - alphas_cumprod
    )

    # (12) in [0]
    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Calculate the mean and variance of the normal distribution q(x_{t-1}|x_t, x_0)

        Use this to go BACKWARDS 1 step, given the current step
        (12) in [0]

        :param x_t: The result of the next step in the diffusion process (batch)
        :param x_start: The initial image (batch)
        :param t: The current timestep (batch)
        """
        mean = (
            extract_for_timesteps(posterior_mean_coef_x_0, t) * x_start
            + extract_for_timesteps(posterior_mean_coef_x_t, t) * x_t
        )
        return mean, extract_for_timesteps(posterior_variance, t)

    return q_posterior_mean_variance


def make_q_sample(alphas_cumprod):
    sqrt_alphas_cumprod = th.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = th.sqrt((1 - alphas_cumprod))

    def q_sample(self, x_0, t, noise):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0)
        Use this to go FORWARDS t steps, given the initial image

        :param x_start: The initial image (batch)
        :param t: The timestep to diffuse to (batch)
        :param noise: The noise (epsilon in the paper)
        """

        N = x_0.shape[0]
        assert t.shape == (N,)

        # (9) in [0]
        mean = extract_for_timesteps(sqrt_alphas_cumprod, t, None) * x_0
        var = extract_for_timesteps(sqrt_one_minus_alphas_cumprod, t, None)
        return mean + var * noise

    return q_sample


def make_p_mean_variance(betas, posterior_variance, alphas_cumprod):
    log_betas = th.log(betas)
    log_posterior_variance = th.log(posterior_variance)
    # TODO: Figure this out
    log_posterior_variance_clipped = log_posterior_variance
    recip_sqrt_alphas_cumprod = 1 / th.sqrt(alphas_cumprod)
    # OpenAI code does:
    #     self.sqrt_recip_alphas_cumprod = np.sqrt(1. / self.alphas_cumprod)
    # which is same as this, since: sqrt(1/a) = 1/sqrt(a)
    sqrt_recip_alphas_cumprod_minus1 = th.sqrt((1 / alphas_cumprod) - 1)

    def p_mean_variance(self, model, x_t, t):
        model_output = model(x_t, t)
        # model_eps: the model is predicting the noise
        # model_v:
        # > our model outputs a vector `v` [...] and we turn this
        # > output into variances
        model_eps, model_v = rearrange(
            model_output, "B (split C) ... -> split B C ...", split=2
        )

        # Turn the model output into a variance (15) in [0]
        min_log = extract_for_timesteps(log_posterior_variance_clipped, t)
        max_log = extract_for_timesteps(log_betas, t)

        # Model outputs between [-1, 1] for [min_var, max_var]
        frac = (model_v + 1) / 2
        pred_var = th.exp(frac * max_log + (1 - frac) * min_log)

        # predict x_0
        # (re-arrange & simplify (9) in [0] to solve for x_0)
        x_0 = (
            x_t * extract_for_timesteps(recip_sqrt_alphas_cumprod, t)
            - extract_for_timesteps(sqrt_recip_alphas_cumprod_minus1, t) * model_eps
        )

        # static thresholding
        x_0 = x_0.clamp(-1, 1)


class GaussianDiffusion:
    """
    Implemented using a bag-of-functions approach
    self.<var_name> is never used in any of the functions

    Pros:
        - It's clear what dependencies each function has
        - Functional style makes it clear no mutation is happening
    Cons:
        - Functions might rederive intermediate dependencies
        - Someone could try to do mutation inside the closure
            (although the lack of `self` signals this would be a bad idea)
    """

    def __init__(self, *, betas):
        bind = lambda f: types.MethodType(f, self, self.__class__)

        # For the cosine schedule, we end up recalculating the alphas
        # but it's more elegant to only be provided the betas
        alphas = 1 - betas
        alphas_cumprod = th.cumprod(alphas)
        # TODO(verify): By prepending 1, the 1st beta is 0
        # This represents the initial image, which as a mean but no variance (since it's ground truth)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        # (10) in [0]
        posterior_variance = ((1 - alphas_cumprod_prev) / (1 - alphas_cumprod)) * betas

        self.q_posterior_mean_variance = bind(
            make_q_posterior_mean_variance(
                alphas, alphas_cumprod, alphas_cumprod_prev, betas, posterior_variance
            )
        )
        self.q_sample = bind(make_q_sample(alphas_cumprod))
        self.p_mean_variance = bind(
            make_p_mean_variance(betas, posterior_variance, alphas_cumprod)
        )

    def vlb_loss_terms(self):
        pass

    def training_loss(self, x_0, t):
        noise = th.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise)

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
