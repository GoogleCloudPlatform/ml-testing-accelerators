local base = import 'base.libsonnet';
local mixins = import '../../mixins.libsonnet';
local timeouts = import "../../timeouts.libsonnet";
local tpus = import '../../tpus.libsonnet';

{
  local resnet = base.GardenTest {
    modelName: 'resnet-ctl',
    command: [
      'python3',
      'official/vision/image_classification/resnet_ctl_imagenet_main.py',
      '--tpu=$(KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS)',
      '--data_dir=gs://imagenet-us-central1/train',
      '--distribution_strategy=tpu',
      '--batch_size=1024',
      '--steps_per_loop=500',
      '--use_synthetic_data=false',
      '--dtype=fp32',
      '--enable_eager=true',
      '--enable_tensorboard=true',
      '--log_steps=50',
      '--single_l2_loss_op=true',
      '--use_tf_function=true',
      '--clean',
    ],
  },
  local functional = mixins.Functional {
    command+: [
      '--train_epochs=1',
      '--epochs_between_evals=1',
    ],
  },
  local convergence = mixins.Convergence {
    accelerator+: tpus.Preemptible,
    command+: [
      '--train_epochs=90',
      '--epochs_between_evals=90',
    ],
  },
  local v2_8 = {
    accelerator+: tpus.v2_8,
    command+: [ '--batch_size=1024' ],
  },
  local v3_8 = {
    accelerator+: tpus.v3_8,
    command+: [ '--batch_size=2048' ],
  },

  configs: [
    resnet + v2_8 + functional,
    resnet + v3_8 + functional,
    resnet + v2_8 + convergence + timeouts.Hours(16),
    resnet + v3_8 + convergence,
  ],
}