local timeouts = import 'timeouts.libsonnet';

{
  Functional:: {
    mode: 'functional',
    timeout: timeouts.one_hour,
    accelerator+: {
      preemptible: true,
    },
  },
  Convergence:: {
    mode: 'convergence',
    timeout: timeouts.ten_hours,
    accelerator+: {
      preemptible: false,
    },
  }
}