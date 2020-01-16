local jobs = import '../jobs.libsonnet';

{
  PyTorchJobConfig:: jobs.PyTorchJobConfig {
    framework_prefix: 'pt-nightly',
    framework_version: 'pytorch-nightly',
    image_tag: 'nightly', 
  },
}