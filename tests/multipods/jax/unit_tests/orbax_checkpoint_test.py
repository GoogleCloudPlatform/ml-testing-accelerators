#!/usr/bin/python3
from absl import flags
import os
import sys
from etils import epath
import jax
from jax.experimental import maps
from jax.experimental.pjit import PartitionSpec
from jax.experimental.pjit import pjit
import numpy as np
import orbax.checkpoint as orbax
import time
#os.system("pip3 install google-cloud-storage") # Called in the libsonnet file instead
from google.cloud import storage

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
os.environ["JAX_USE_PJRT_C_API_ON_TPU"] = "1"

print(
    f"orbax_checkpoint_test: Devices found. number {len(jax.devices())}, details: {jax.devices()}",
    flush=True,
)

FLAGS = flags.FLAGS

bucket_path_flag = flags.DEFINE_string(
    "bucket_path", None, "GCS bucket path, e.g. gs://my-bucket")

dir_name_flag = flags.DEFINE_string(
    "ckpt_dir", None, "GCS cloud bucket directory e.g. orbax-checkpoints")

FLAGS(sys.argv)  # parses the flags.

bucket_path = epath.Path(bucket_path_flag.value)
dir_name = dir_name_flag.value

mngr = orbax.CheckpointManager(
    ckpt_dir, orbax.Checkpointer(orbax.PyTreeCheckpointHandler()))

print("orbax_checkpoint_test: Created CheckpointManager")

BATCH = 256
SIZE = 1024

def gen_data():
  return jax.process_index() * jax.numpy.ones(
      (BATCH, SIZE, SIZE), dtype=jax.numpy.bfloat16)

# Define array lengths based on device count
num_devices = len(jax.devices())
mesh_shape = [len(jax.devices())]
devices = np.asarray(jax.devices()).reshape(*mesh_shape)
mesh = maps.Mesh(devices, ("x"))

gen_data_sharded = pjit(
    gen_data, in_axis_resources=None, out_axis_resources=PartitionSpec("x")
)

with maps.Mesh(mesh.devices, mesh.axis_names):
  presharded_X = jax.block_until_ready(gen_data_sharded())
  to_save = {"my_X": presharded_X}
  print("orbax_checkpoint_test: Attempting save...")
  mngr.save(0, to_save)
  print("orbax_checkpoint_test: Save successful!")

print("orbax_checkpoint_test: Attempting restore...!")
s2 = mngr.restore(0)
print("orbax_checkpoint_test: Restore successful!")

print("orbax_checkpoint_test: Test finished successfully")
