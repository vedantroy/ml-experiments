import types

import torch as th
import torch.nn.functional as F
from einops import rearrange
import math

# TODO: Re-implement these
from losses import discretized_gaussian_log_likelihood, normal_kl
from nn import mean_flat

# Simple-ish Gaussian Diffusion
# Notes:
# - Learns the variance
#  - Scales the variance loss to avoid L_vlb overwhelming L_simple
#  - Applies a stop-gradient to the mean term for L_vlb to only allow
#    backpropagation through the variance term
# - Estimates the noise (epsilon)
# - Uses a hybrid objective (L_simple + L_vlb) without resampling

# Question list:
# 1. the clamp seems unnecessary?
# 2. Why do we set 1 as the 1st alpha value? I notice this makes the 1st Beta be 0
# are the betas indexed s.t betas[0] == represents the image before any diffusion process?
# (this would make sense b/c if the 1st beta is 0, then there would be no variance)


def cosine_betas(timesteps, s=0.008, max_beta=0.999):
    """
    Get B_t for the cosine schedule (eq 17 in [0])

    :param max_beta: "In practice, we clip B_t to be no larger than 0.999 to prevent
                      singularities at the end of the diffusion process near t = T"
    :param s: "We use a small offset s to prevent B_t from being too small near t = 0"
    """
    # If we add noise twice, then there are 3 total states (0, 1, 2)
    states = timesteps + 1
    t = th.linspace(start=0, end=timesteps, steps=states, dtype=th.float64)
    f_t = th.cos(((t / timesteps) + s) / (1 + s) * (math.pi / 2)) ** 2
    alphas_cumprod = f_t / f_t[0]
    alphas_cumprod_t = alphas_cumprod[1:]
    alphas_cumprod_t_minus_1 = alphas_cumprod[:-1]
    betas = 1 - (alphas_cumprod_t / alphas_cumprod_t_minus_1)
    # TODO: In practice, this clamp just seems to clip the last value from 1 to 0.999
    return betas.clamp(0, max_beta)


def extract_for_timesteps(x, timesteps, broadcast_shape):
    # This will most certainly need to be fixed later
    vals = x[timesteps]
    return vals


class GaussianDiffusion:
    def __init__(self, betas):
        self.n_timesteps = betas.shape[0]
        alphas = 1 - betas
        alphas_cumprod = th.cumprod(alphas, dim=0)

        # TODO(verify): By prepending 1, the 1st beta is 0
        # This represents the initial image, which as a mean but no variance (since it's ground truth)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        # (10) in [0]
        self.posterior_variance = (
            (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
        ) * betas

        # (11) in [0]
        self.posterior_mean_coef_x_0 = (th.sqrt(alphas_cumprod_prev) * betas) / (
            1 - alphas_cumprod
        )
        self.posterior_mean_coef_x_t = (alphas.sqrt() * (1 - alphas_cumprod_prev)) / (
            1 - alphas_cumprod
        )

        self.sqrt_alphas_cumprod = th.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = th.sqrt((1 - alphas_cumprod))

        # Used to calculate the variance
        self.log_betas = th.log(betas)

        # clipped to avoid log(0) == -inf
        # In other words, posterior variance is 0 at the start of the diffusion chain
        assert alphas_cumprod_prev[0] == 1 and self.posterior_variance[0] == 0

        self.posterior_log_variance_clipped = th.log(
            F.pad(self.posterior_variance[1:], (1, 0), value=self.posterior_variance[1])
        )

        self.recip_sqrt_alphas_cumprod = 1 / th.sqrt(alphas_cumprod)
        # OpenAI code does:
        #     self.sqrt_recip_alphas_cumprod = np.sqrt(1. / self.alphas_cumprod)
        # which is same as this, since: sqrt(1/a) = 1/sqrt(a)
        self.sqrt_recip_alphas_cumprod_minus1 = th.sqrt((1 / alphas_cumprod) - 1)
        # assert "figure out log var clipping" and False

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
            extract_for_timesteps(self.posterior_mean_coef_x_0, t) * x_start
            + extract_for_timesteps(self.posterior_mean_coef_x_t, t) * x_t
        )
        return mean, extract_for_timesteps(self.posterior_variance, t)

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
        mean = extract_for_timesteps(self.sqrt_alphas_cumprod, t, None) * x_0
        var = extract_for_timesteps(self.sqrt_one_minus_alphas_cumprod, t, None)
        return mean + var * noise

    def predict_x0_from_eps(self, x_t, t, eps):
        # predict x_0
        # (re-arrange & simplify (9) in [0] to solve for x_0)
        return (
            x_t * extract_for_timesteps(self.recip_sqrt_alphas_cumprod, t)
            - extract_for_timesteps(self.sqrt_recip_alphas_cumprod_minus1, t) * eps
        )

    def model_v_to_log_variance(self, v, t):
        # Turn the model output into a variance (15) in [0]
        min_log = extract_for_timesteps(self.log_posterior_variance_clipped, t)
        max_log = extract_for_timesteps(self.log_betas, t)

        # Model outputs between [-1, 1] for [min_var, max_var]
        frac = (v + 1) / 2
        return frac * max_log + (1 - frac) * min_log

    def vb_loss(self, *, x_0, true_mean, true_log_var, pred_mean, pred_log_var, t):
        kl = normal_kl(true_mean, true_log_var, pred_mean, pred_log_var)
        kl = mean_flat(kl) / math.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_0, means=pred_mean, log_scales=0.5 * pred_log_var
        )
        decoder_nll = mean_flat(decoder_nll) / math.log(2.0)

        # `th.where` selects from tensor 1 if cond is true and tensor 2 otherwise
        return th.where((t == 0), decoder_nll, kl)

    def p_mean_variance(self, x_t, t, model_v, model_eps, *, threshold):
        # calculate x_0 from the predicted noise & use it to calculate
        # the estimated mean and variance
        pred_x_0 = self.predict_x0_from_eps(x_t, t, model_eps)
        if threshold:
            # Question: Why do we apply clamping before q_posterior?
            pred_x_0 = pred_x_0.clamp(-1, 1)
        pred_mean, _, _ = self.q_posterior_mean_variance(pred_x_0, x_t, t)
        pred_log_var = self.model_v_to_log_variance(model_v, t)
        return pred_mean, pred_log_var

    def training_losses(self, model, x_0, t):
        noise = th.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise)

        C = x_0.shape[1]
        model_output = model(x_t, t)

        # model_eps: the model is predicting the noise
        # model_v:
        # > our model outputs a vector `v` [...] and we turn this
        # > output into variances
        # from [0]
        model_eps, model_v = rearrange(
            model_output, "B (split C) ... -> split B C ...", split=2, C=C
        )
        mse_loss = mean_flat((noise - model_eps) ** 2)

        # calculate the variational lower bound
        true_mean, _, true_log_var_clipped = self.q_posterior_mean_variance(x_0, x_t, t)

        pred_mean, pred_log_var = self.p_mean_variance(
            x_t, t, model_v, model_eps, threshold=False
        )

        # > Along this same line of reasoning,
        # > we also apply a stop-gradient to the µθ(xt, t) output for the
        # > L_vlb term. This way, Lvlb can guide Σθ(xt, t) while L_simple
        # > is still the main source of influence over µθ(xt, t)
        # from [0]
        frozen_mean = true_mean.detach()
        vb_loss = self.vb_loss(
            frozen_mean, true_log_var_clipped, pred_mean, pred_log_var
        )

        # > For our experiments, we set λ = 0.001 to prevent L_vlb from
        # > overwhelming L_simple
        # from [0]
        vb_loss *= self.n_timesteps / 1000.0
        return mse_loss + vb_loss
