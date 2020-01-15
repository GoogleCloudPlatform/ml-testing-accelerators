local jobs = import '../jobs.libsonnet';

{
  GardenJobConfig:: jobs.GardenJobConfig {
    framework_prefix: 'tf-nightly',
    framework_version: 'nightly-2.x',
    image_tag: 'nightly',
  },
}
