local base = import "../base.libsonnet";

{
  GardenTest:: base.BaseTest {
    local config = self,

    image: "gcr.io/xl-ml-test/model-garden",

    jobSpec+:: {
      template+: {
        spec+: {
          containerMap+: {
            train+: {
              args+: [ "--model_dir=$(MODEL_DIR)" ]
            },
          },
        },
      },
    },
  },
}
