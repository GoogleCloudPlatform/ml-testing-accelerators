local common = import 'common.libsonnet';
local timeouts = import 'templates/timeouts.libsonnet';
local tpus = import 'templates/tpus.libsonnet';
local utils = import 'templates/utils.libsonnet';

{
  local colab = common.PyTorchTest + timeouts.Minutes(15) {
    local config = self,

    mode: 'colab',

    accelerator: tpus.v3_8,
    cpu: 8,
    memory: '35Gi',

    colabSettings:: {
      notebook: error 'Must set `colabSettings.notebook`',
    },
    command: utils.scriptCommand(
      |||
        cd /
        git clone https://github.com/pytorch/xla.git

        export COLAB_TPU_ADDR=${KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS#grpc://}
        export TPU_NAME=${TPU_NAME#*/}

        notebook_path='/xla/contrib/colab/%(notebook)s'
        # HACK: Remove usage of google.colab.patches.cv2_imshow
        sed -i '/cv2/d' $notebook_path
        jupyter nbconvert --to notebook --inplace --execute "${notebook_path}"

        gsutil cp "${notebook_path}" "$(MODEL_DIR)/%(notebook)s"
      ||| % config.colabSettings
    ),
  },

  local mnist = {
    modelName: 'mnist',
    colabSettings+:: {
      notebook: 'mnist-training.ipynb',
    },
  },
  local resnet18 = {
    modelName: 'resnet18',
    colabSettings+:: {
      notebook: 'resnet18-training.ipynb',
    },
  },

  configs: [
    colab + mnist,
    colab + resnet18,
  ],
}
