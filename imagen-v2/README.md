## Python Setup
```
conda create -n imagen python=3.9
pip install -r /path/to/requirements.txt
```

## GPU Info
### V100
- 16 GB memory
- Max batch size = 8??
- Although batch through put decreases, overall throughput increases w/ greater batch size
- You don't need 8 CPUs. Use less.
- fp16=True causes NaN loss unless you increase the gradient_update_size
- 1.25 * 8 = 10 imgs/second
- 86400 * 10 = 864,000 imgs/day
    - ~ 6 days / 5 million imgs
