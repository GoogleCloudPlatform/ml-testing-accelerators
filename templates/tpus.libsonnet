{
  TpuSpec:: {
    local tpu = self,

    version: error "Must override `type`",
    size: error "Must override `size`",
    preemptible: false,
    preemptible_prefix:: if tpu.preemptible then
      "preemptible-"
    else
      "",

    name: "v%(version)d-%(size)d" % tpu,
    resource:: "cloud-tpus.google.com/%(preemptible_prefix)sv%(version)s" % tpu,
    resource_limits:: { [tpu.resource]: tpu.size },
  },
  Preemptible:: {
    preemptible: true
  },
  v2_8: self.TpuSpec {version: 2, size: 8},
  v3_8: self.TpuSpec {version: 3, size: 8},
}