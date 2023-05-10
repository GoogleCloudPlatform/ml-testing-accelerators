local common = import 'common.libsonnet';
local experimental = import '../experimental.libsonnet';
local tpus = import 'templates/tpus.libsonnet';

{
  local accelerate = self.accelerate,
  accelerate:: common.PyTorchTest + common.Functional {
    modelName: 'accelerate',
    mode: 'smoke',
    command: [
      'accelerate',
      'test',
    ],
  },
  local tpuVm = self.tpuVm,
  tpuVm:: common.PyTorchTpuVmMixin + experimental.PjRt {
    local config = self,
    tpuSettings+: {
      tpuVmExports+: |||
        export XLA_USE_BF16=$(XLA_USE_BF16)
        export PATH=~/.local/bin:$PATH
      |||,
      tpuVmExtraSetup: |||
        git clone https://github.com/huggingface/accelerate.git
        pip install -e accelerate

        mkdir -p ~/.cache/huggingface/accelerate/
        cat > ~/.cache/huggingface/accelerate/default_config.yaml << 'HF_CONFIG_EOF'
        compute_environment: LOCAL_MACHINE
        distributed_type: TPU
        downcast_bf16: 'no'
        machine_rank: 0
        main_training_function: main
        mixed_precision: 'no'
        num_machines: 1
        num_processes: %d
        rdzv_backend: static
        same_network: true
        tpu_env: []
        tpu_use_cluster: false
        tpu_use_sudo: false
        use_cpu: false
        HF_CONFIG_EOF

        accelerate env
      ||| % [config.accelerator.numCores],
    },
  },
  local v2_8 = self.v2_8,
  v2_8:: {
    accelerator: tpus.v2_8,
  },
  local v4_8 = self.v4_8,
  v4_8:: {
    accelerator: tpus.v4_8,
  },

  configs: [
    accelerate + v2_8 + tpuVm,
    accelerate + v4_8 + tpuVm,
  ],
}
