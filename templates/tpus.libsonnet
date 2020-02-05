{
  TpuSpec:: {
    local tpu = self,
    
    name: "v%(version)d-%(size)d" % tpu,
    version: error "Must override `type`",
    size: error "Must override `size`",
    preemptible: false,

    PodSpec:: {
      containerMap+: {
        train+: {
          resources+: {
            limits+: { [tpu.resource]: tpu.size },
          },
        },
      },
    },

    preemptiblePrefix:: if tpu.preemptible then
      "preemptible-"
    else
      "",
    resource:: "cloud-tpus.google.com/%(preemptiblePrefix)sv%(version)s" % tpu,
  },
  Preemptible:: {
    preemptible: true
  },

  v2_8: self.TpuSpec { version: 2, size: 8 },
  v3_8: self.TpuSpec { version: 3, size: 8 },
}