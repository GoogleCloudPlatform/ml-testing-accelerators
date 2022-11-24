from praxis import py_utils
import jax

if __name__ == '__main__':
  print('Syncing %s global devices' % jax.device_count())
  # Block all hosts until directory is ready.
  py_utils.sync_global_devices('sync devices')
  print('Sync finished')
  exit 0
