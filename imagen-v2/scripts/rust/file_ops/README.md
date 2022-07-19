## Why?
On computers with NVME SSDs, I figured single-threaded copying is not the fastest. So, I wrote this Rust script to copy with all cores.

`htop` shows that the CPUs are barely being utilized anyways ... but it's still much faster than `cp` (which is single-threaded, I think)

I originally wrote this in Rust because I wanted to leverage io_uring, but it seems like there are no good crates for that / it would require Linux kernel version 5.6+ which is not common.

**Lesson:** Don't write Rust unless you're sure you can utilize things like io_uring to massively boost performance.

## Copy Benchmark
### System Config
- 64 cores
- 64 gigs of memory
### Task Config
Copy 58 gigs of files
---
### Results
| Program | Real   | Sys   |
|---------|--------|-------|
| Rust    | 3m58s  | 2m16s |
| Python  | 4m24s  | 2m27s |
| `cp -r` | 13m38s | 1m44s |