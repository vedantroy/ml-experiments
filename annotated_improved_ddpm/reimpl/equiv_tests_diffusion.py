from openai_diffusion import get_named_beta_schedule
from diffusion import cosine_betas

import torch as th
from torch import testing

def test_cosine_betas():
    T = 4000
    betas1 = get_named_beta_schedule("cosine", T)
    betas2 = cosine_betas(T)
    testing.assert_close(th.from_numpy(betas1), betas2)

test_cosine_betas()