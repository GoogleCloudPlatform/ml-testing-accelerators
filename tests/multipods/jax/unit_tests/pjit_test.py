#!/usr/bin/python3
import jax
from jax.experimental.pjit import PartitionSpec, pjit
import numpy as np
from jax.experimental import maps
import datetime
import os


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
os.environ["JAX_USE_PJRT_C_API_ON_TPU"] = "1"

print(f"devices found. number {len(jax.devices())}, details: {jax.devices()}", flush = True)

def simple_timeit(f, tries = 10, verbose = True):
  outcomes = []
  f() #warm it up!
  for i in range(tries):
    s = datetime.datetime.now()
    r = f()
    e = datetime.datetime.now()
    outcomes.append((e-s).total_seconds())
  average_time = sum(outcomes)/len(outcomes)
  if verbose:
    print(f"average time: {average_time}, timings (seconds) {outcomes}")
  return average_time

BATCH = 1024
SIZE = 1024

def training_loop(inp):
  return jax.numpy.sum(inp, axis = 2)

def gen_data():
  return jax.numpy.ones((BATCH, SIZE, SIZE), dtype=jax.numpy.bfloat16 )

mesh_shape = [len(jax.devices())]
devices = np.asarray(jax.devices()).reshape(*mesh_shape)
mesh = maps.Mesh(devices, ('x'))

pjit_training_loop = pjit(
        training_loop,
        in_axis_resources=PartitionSpec('x'),
        out_axis_resources=PartitionSpec(None)
      )

gen_data_sharded = pjit(
        gen_data,
        in_axis_resources=None,
        out_axis_resources=PartitionSpec('x')
      )


with maps.Mesh(mesh.devices, mesh.axis_names):
  presharded_X = jax.block_until_ready(gen_data_sharded())
  print(f'presharded_X shape {presharded_X.shape}')
  total = jax.block_until_ready(pjit_training_loop(presharded_X))
  np.testing.assert_array_equal(total, np.full((1024, 1024), 1024))
  time = simple_timeit(lambda : jax.block_until_ready(pjit_training_loop(presharded_X)))
