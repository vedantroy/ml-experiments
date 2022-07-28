from openai_diffusion import (
    get_named_beta_schedule,
    GaussianDiffusion,
    ModelMeanType,
    ModelVarType,
    LossType,
)
from diffusion import cosine_betas, GaussianDiffusion as MyGaussianDiffusion

import torch as th
from torch import testing


def test_cosine_betas():
    T = 4000
    betas1 = get_named_beta_schedule("cosine", T)
    betas2 = cosine_betas(T)
    testing.assert_close(th.from_numpy(betas1), betas2)
    print("test_cosine_betas passed")

test_cosine_betas()

def test_gaussian_diffusion_vars():
    T = 4000
    betas = cosine_betas(T)
    gd = GaussianDiffusion(
        betas=betas,
        model_mean_type=ModelMeanType.EPSILON,
        model_var_type=ModelVarType.LEARNED_RANGE,
        # this represents the MSE loss + VLB loss w/ VLB loss scaled down by 1000
        loss_type=LossType.RESCALED_MSE,
        rescale_timesteps=False,
    )

    my_gd = MyGaussianDiffusion(betas)
    testing.assert_close(th.from_numpy(gd.posterior_variance), my_gd.posterior_variance)
    testing.assert_close(th.from_numpy(gd.posterior_mean_coef1), my_gd.posterior_mean_coef_x_0)
    testing.assert_close(th.from_numpy(gd.posterior_mean_coef2), my_gd.posterior_mean_coef_x_t)
    testing.assert_close(th.from_numpy(gd.posterior_log_variance_clipped), my_gd.posterior_log_variance_clipped)
    print("test_gaussian_diffusion_vars passed")

test_gaussian_diffusion_vars()