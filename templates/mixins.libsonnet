local timeouts = import "timeouts.libsonnet";

{
  Functional:: {
    mode: "functional",
    timeout: timeouts.one_hour,
    schedule: "0 */12 * * *",
    accelerator+: {
      preemptible: true,
    },
  },
  Convergence:: {
    mode: "convergence",
    timeout: timeouts.ten_hours,
    schedule: "30 7 * * */2",
    accelerator+: {
      preemptible: false,
    },
  }
}