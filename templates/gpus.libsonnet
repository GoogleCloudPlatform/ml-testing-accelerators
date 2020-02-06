{
  GPUSpec:: {
    local gpu = self,

    name: "%(type)s-x%(number)d" % gpu,
    type: error "Must specify GPUSpec `type`",
    number: 1,

    PodSpec:: {
      containerMap+: {
        train+: {
          resources+: {
            limits+: {
              "nvidia.com/gpu": gpu.number
            },
          },
        },
      },
      nodeSelector+: {
        "cloud.google.com/gke-accelerator": "nvidia-%(type)s" % gpu,
      },
    },
  },

  teslaV100: self.GPUSpec { type: "tesla-v100" },
}
