## Python Setup
```
conda create -n imagen python=3.9
pip install -r /path/to/requirements.txt
```

## GPU Batch Sizes
### V100
- 16 GB memory
- Max batch size = 8??
- Although batch through put decreases, overall throughput increases w/ greater batch size
- You don't need 8 CPUs. Use less.