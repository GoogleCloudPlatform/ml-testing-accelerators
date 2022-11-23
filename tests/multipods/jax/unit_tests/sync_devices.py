from jax.experimental import multihost_utils

if __name__ == '__main__':
  print('Syncing %s global devices', jax.device_count())
  multihost_utils.sync_global_devices('sync')
  exit()

