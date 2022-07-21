I extracted the model from [Improved Denoising Diffusion Probabilistic Models](https://github.com/openai/improved-diffusion) and made it easy to run as a standalone script.

Objective is to assert all the tensor shapes and not have the model crash

## Notes for Original Repo
This is good for
Installing mpi4py is hard in the original repo. Just use `sudo`.
On Ubuntu you'll also need to install the dev lib

Use this script when tinkering with the original repo:
```
pip uninstall improved_diffusion -y
pip install -e .
MODEL_FLAGS="--image_size 64 --num_channels 128 --num_res_blocks 3"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule linear"
TRAIN_FLAGS="--lr 1e-4 --batch_size 2"
python3 scripts/image_train.py --data_dir cifar_train $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
```