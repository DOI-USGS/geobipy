# -*- coding: utf-8 -*-
import pickle
from numpy.random import Generator, PCG64DXSM


def get_prng(generator=PCG64DXSM, seed=None, jump=None, world=None):
    """Generate an independent prng.

    Returns
    -------
    seed : int or file, optional
        The seed of the bit generator.
    jump : int, optional
        Jump the bit generator by this amount
    world : mpi4py.MPI.COMM_WORLD, optional
        MPI communicator, will jump each bit generator by world.rank

    """
    # Default to single core, else grab the mpi rank.
    rank = 0
    if world is not None:
        rank = world.rank

    if rank == 0:
        if seed is not None: # Seed is a file.
            if isinstance(seed, str):
                with open(seed, 'rb') as f:
                    seed = pickle.load(f)
            assert isinstance(seed, int), TypeError("Seed {} must have type python int (not numpy)".format(seed))

        else: # No seed, generate one
            bit_generator = generator()
            seed = bit_generator._seed_seq.entropy
            with open('seed.pkl', 'wb') as f:
                pickle.dump(seed, f)
            print('Seed: {}'.format(seed), flush=True)

    if world is not None:
        # Broadcast the seed to all ranks.
        seed = world.bcast(seed, root=0)

    bit_generator = generator(seed = seed)

    if world is not None:
        jump = world.rank

    if jump is not None:
        bit_generator = bit_generator.jumped(jump)

    return Generator(bit_generator)