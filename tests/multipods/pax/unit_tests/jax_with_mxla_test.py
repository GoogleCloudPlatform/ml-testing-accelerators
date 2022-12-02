import jax
from jax.experimental import pjit
from jax.experimental.maps import Mesh
import jax.numpy as jnp
import numpy as np

_PartitionSpec = jax.experimental.PartitionSpec


def test_one_plus_one() -> None:
  """Test one plus one."""

  res = jax.numpy.add(1, 1)
  assert res == 2
  print("test_one_plus_one is done.")


def test_mxla_psum() -> None:
  """Test MXLA psum."""

  num_slices = 2
  num_devices = jax.device_count()
  num_local_devices = jax.local_device_count()
  numpy_add_res = jnp.add(1, 1)

  slice_index = jax.local_devices()[0].slice_index
  num_devices_per_slice = num_devices / num_slices
  expected_value = np.sum(np.arange(num_slices)) * num_devices_per_slice

  def fn(v):
    return jax.lax.psum(v, axis_name="i")

  shape = [num_local_devices, 32, 1024]
  expected = np.full(shape, expected_value)
  x = np.full(shape, slice_index)
  y = jax.pmap(fn, devices=jax.devices(), axis_name="i")(x)
  np.testing.assert_array_equal(np.array(y), expected)
  print("test_mxla_psum is done.")


def test_megascale_pjit():
  """Test MegaScale pjit."""

  devices = jax.devices()

  def fun(x):
    return x, x.sum()

  out_axis_resources = None
  in_axis_resources = _PartitionSpec(("data", "model"))
  fn = pjit.pjit(fun, in_axis_resources, out_axis_resources)

  with Mesh(
      np.array(devices).reshape((2, len(devices) // 2)), ("model", "data")):
    ids, id_sum = fn(np.array([d.id for d in jax.local_devices()]))
    np.testing.assert_array_equal(
        ids,
        np.array(
            np.array([d.id for d in devices]).reshape(
                (2, len(devices) // 2)).transpose().flat))
    np.testing.assert_array_equal(id_sum,
                                  np.array([d.id for d in jax.devices()]).sum())
  print("test_megascale_pjit is done.")


def test_jit() -> None:
  """Test jit."""

  np.testing.assert_array_equal(jnp.arange(4), np.arange(4))
  print("test_jit is done.")


def test_megascale_device_copy() -> None:
  """Test MegaScale device copy."""

  x = np.arange(4)
  assert len(jax.local_devices()) > 1
  for d in jax.local_devices():
    x = jax.jit(lambda x: x, device=d)(x)
  np.testing.assert_array_equal(x, np.arange(4))
  print("test_megascale_device_copy is done.")


test_one_plus_one()
test_mxla_psum()
test_megascale_pjit()
test_jit()
test_megascale_device_copy()

