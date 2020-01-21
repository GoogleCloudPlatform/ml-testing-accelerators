local jobs = import 'jobs.libsonnet';
local modes = import '../../modes.libsonnet';
local timeouts = import "../../timeouts.libsonnet";
local tpus = import '../../tpus.libsonnet';

{
  local resnet = jobs.GardenJobConfig {
    model_name: 'resnet-cfit',
    command: [
      'python3',
      'official/vision/image_classification/resnet_imagenet_main.py',
      '--tpu=$(KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS)',
      '--data_dir=gs://imagenet-us-central1/train',
      '--batch_size=1024',
      '--steps_per_loop=500',
      '--skip_eval=false',
      '--use_synthetic_data=false',
      '--dtype=fp32',
      '--enable_eager=true',
      '--enable_tensorboard=true',
      '--enable_checkpoint_and_export',
      '--distribution_strategy=tpu',
      '--report_accuracy_metrics=true',
      '--clean',
    ],
  },
  local functional = modes.Functional {
    command+: [
      '--train_epochs=1',
      '--epochs_between_evals=1',
    ],
  },
  local convergence = modes.Convergence {
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
    resnet + v2_8 + convergence + timeouts.hours(16),
    resnet + v3_8 + convergence,
  ],
}