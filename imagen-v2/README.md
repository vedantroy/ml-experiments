## Python Setup
```
conda create -n imagen python=3.9
pip install -r /path/to/requirements.txt
```

This should be handled by requirements.txt, but to make sure
(run this inside the conda venv):
```
pip uninstall Pillow
pip uninstall PIL
Pip install Pillow-SIMD
```

then try
```
import PIL
print(PIL.__version__)
```

if there's a postfix, you are using Pillow-SIMD
Interestingly, Pillow-SIMD might make the pre-processing scripts slower ??

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

## Lessons / Dev Log
- Don't bother with fancy systems programming. Stuff like io_uring is fun
but definitely useless b/c bottlenecks are always processing / running models.

Besides, you should only be doing reading => doing *all* processing => writing.
io_uring is more useful if you're doing lots of reads/writes, which you shouldn't
