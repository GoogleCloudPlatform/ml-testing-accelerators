local base = import '../base.libsonnet';

{
  GardenTest:: base.GardenTest {
    framework_prefix: 'tf-nightly',
    framework_version: 'nightly-2.x',
    image_tag: 'nightly',
  },
}
