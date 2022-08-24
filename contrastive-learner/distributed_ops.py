import argparse

import torch as th
import torch.distributed as dist

if __name__ == "__main__":
	argparse = argparse.ArgumentParser()
	argparse.add_argument("rank", type=int)
	args = argparse.parse_args()

	dist.init_process_group(
		backend="nccl", 
		init_method="tcp://localhost:8001",
		rank=args.rank,
		# Testing on 2 GPUs
		world_size=2
	)

	rank = dist.get_rank()
	assert rank == args.rank

	device = th.device(f"cuda:{rank}")

	# For NCCL, the tensors must be on different GPUs
	tensor_len = 3
	vals = [(rank * tensor_len) + i for i in range(tensor_len)]
	tensor = th.IntTensor(vals).to(device=device)
	# tensor = th.full((3,), rank).to(device=device)
	print(f"Rank: {rank}")
	print(f"Start tensor: {tensor}")
	output = list(th.empty_like(tensor) for _ in range(dist.get_world_size()))
	# print(f"All-gather list: {output}")
	dist.all_gather(output, tensor)

	print(f"Gathered: {output}")
	
	# catted = th.cat(output)
	scatter_size = 5
	aggregate = th.IntTensor(list(range(dist.get_world_size() * scatter_size))).to(device=device)
	input_list = list(aggregate.chunk(dist.get_world_size()))
	print(f"Input list: {input_list}")
	rscatter_input = th.empty_like(input_list[dist.get_rank()])
	dist.reduce_scatter(rscatter_input, input_list)
	print(f"Reduce Scatter output: {rscatter_input}")

	aggregate = th.IntTensor([dist.get_rank() * scatter_size + i for i in range(dist.get_world_size() * scatter_size)]).to(device=device)
	input_list = list(aggregate.chunk(dist.get_world_size()))
	print(f"Input list 2: {input_list}")
	rscatter_input = th.empty_like(input_list[dist.get_rank()])
	dist.reduce_scatter(rscatter_input, input_list)
	print(f"Reduce Scatter output 2: {rscatter_input}")
