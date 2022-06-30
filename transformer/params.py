params = {
    # Model params
    "sequence_len": 64,
    ## Length of an embedding vector
    "d_model": 128,
    # original paper, d_ff = 2048, d_model = 512; 2048/512 = 4
    "widening_factor": 5,
    "heads": 4,
    "layers": 4,
    # Training params
    "batch_size": 2,
    "epochs": 2,
}

constants = {"vocab_size": 128}
