local base = import '../base.libsonnet';

{
  PyTorchTest:: base.PyTorchTest {
    framework_prefix: 'pt-nightly',
    framework_version: 'pytorch-nightly',
    image_tag: 'nightly', 
  },
}