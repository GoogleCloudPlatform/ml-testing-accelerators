


@experiment_registry.register
class LmCloudSpmdMultislice2BXLML(LmCloudSpmd):
  """SPMD model with 2B params on 2x 2x2x1 slices.
  """
  PERCORE_BATCH_SIZE = 2

  NUM_LAYERS = 18
  MODEL_DIMS = 3072
  HIDDEN_DIMS = MODEL_DIMS * 4

  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_NOTHING
  # 2-way replica parallelism on DCN and 4-way data parallelism on ICI.
  ICI_MESH_SHAPE = [1, 4, 1]
  DCN_MESH_SHAPE = [2, 1, 1]
  def task(self) -> tasks_lib.SingleTask.HParams:
    task_p = super().task()
    task_p.train.num_train_steps = 50
    return task_p

