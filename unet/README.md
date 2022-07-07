# Carvana UNet

This is a UNet I trained on the Carvana dataset
Code is heavily based off of: https://github.com/milesial/Pytorch-UNet/

The code is super messy, but it works pretty well.
For some reason, using AMP makes the model slower?

lightning_train.py uses Pytorch Lightning instead, but I gave up finishing
the implementation because I didn't see the benefit of using Pytorch Lightning