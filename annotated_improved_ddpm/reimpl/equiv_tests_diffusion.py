from openai_diffusion import (
    get_named_beta_schedule,
    GaussianDiffusion,
    ModelMeanType,
    ModelVarType,
    LossType,
)
from diffusion import cosine_betas, GaussianDiffusion as MyGaussianDiffusion, extract_for_timesteps

import torch as th
from torch import testing

# throwing in a test for `torch.gather` intuition
def test_torch_gather():
    x = th.tensor([1, 2, 3])
    vals = x.gather(0, th.tensor([0, 2]))
    testing.assert_close(vals, th.tensor([1, 3]))
    print("test_torch_gather passed")

test_torch_gather()

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
    f32 = lambda x: th.from_numpy(x).to(th.float32)
    # testing.assert_close(f32(gd.posterior_variance), my_gd.posterior_variance)
    testing.assert_close(f32(gd.posterior_mean_coef1), my_gd.posterior_mean_coef_x_0)
    testing.assert_close(f32(gd.posterior_mean_coef2), my_gd.posterior_mean_coef_x_t)
    testing.assert_close(f32(gd.posterior_log_variance_clipped), my_gd.posterior_log_variance_clipped)

    # predict_x0 tests
    testing.assert_close(f32(gd.sqrt_recip_alphas_cumprod), my_gd.recip_sqrt_alphas_cumprod)
    testing.assert_close(f32(gd.sqrt_recipm1_alphas_cumprod), my_gd.sqrt_recip_alphas_cumprod_minus1)

    print("test_gaussian_diffusion_vars passed")


test_gaussian_diffusion_vars()

def test_gaussian_diffusion_funcs():
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

    N, C, H, W = 2, 3, 64, 64
    x_0 = th.randn((N, C, H, W))
    noise = th.randn_like(x_0)
    t =  th.tensor([295, 2253])

    x_t = gd.q_sample(x_0, t, noise)
    my_x_t = my_gd.q_sample(x_0, t, noise)
    testing.assert_close(x_t, my_x_t)

    x_tm1, _, log_var = gd.q_posterior_mean_variance(x_0, x_t, t)
    my_x_tm1 = my_gd.q_posterior_mean(x_0, x_t, t)
    my_log_var = extract_for_timesteps(my_gd.posterior_log_variance_clipped, t, x_tm1.shape)

    # my extract_for_timesteps method returns a different shape
    # (relies on broadcasting instead of expanding dims)
    assert (my_log_var == log_var).all()
    testing.assert_close(x_tm1, my_x_tm1)

    fake_model_out = th.randn((N, C * 2, H, W))
    out = gd.p_mean_variance(lambda *args, r=fake_model_out: r, x_t, t, clip_denoised=False)
    pred_mean, pred_var = out['mean'], out['log_variance']

    model_eps, model_v = th.chunk(fake_model_out, 2, dim=1)
    my_pred_mean, my_pred_var = my_gd.p_mean_variance(x_t, t, model_v, model_eps, threshold=False)

    # to prove it's actually working!
    # pred_var[0][0][0][0] += 1e-3
    testing.assert_close(pred_var, my_pred_var)
    testing.assert_close(pred_mean, my_pred_mean)

    print("test_gaussian_diffusion_funcs passed")

test_gaussian_diffusion_funcs()

def test_gaussian_diffusion_e2e():
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

    N, C, H, W = 2, 3, 64, 64

    x_0 = th.randn((N, C, H, W))
    noise = th.randn_like(x_0)
    t =  th.tensor([295, 2253])

    fake_output = th.randn((N, C * 2, H, W))

    model = lambda *args, r=fake_output: r

    losses = gd.training_losses(model, x_0, t, noise=noise)

    x_t = my_gd.q_sample(x_0, t, noise)
    my_losses = my_gd.training_losses(fake_output, x_0, x_t, t, noise)

    testing.assert_close(losses["loss"], my_losses)
    print("test_gaussian_diffusion_e2e passed")

test_gaussian_diffusion_e2e()