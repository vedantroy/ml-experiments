# Image Loading Profiling

How time is spent reading the image into `PIL` vs. converting the `PIL` image into a tensor

In the benchmarks:
- dark blue = python
- medium blue = native
- light blue = system

A significant portion of time is spent in Python code, which we can eliminate if we pre-process the images into tensors