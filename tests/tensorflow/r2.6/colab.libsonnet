local common = import 'common.libsonnet';
local tpus = import 'templates/tpus.libsonnet';
local utils = import 'templates/utils.libsonnet';

{
  local colab = common.ModelGardenTest {
    local config = self,

    mode: 'colab',
    schedule: common.Functional.schedule,
    accelerator: tpus.v3_8,

    colabSettings:: {
      notebook: error 'Must set `colabSettings.notebook`',
    },
    command: utils.scriptCommand(
      |||
        cd /
        git clone https://github.com/tensorflow/tpu.git
        pip install jupyter

        apt-get install -y wget
        mkdir /content

        export COLAB_TPU_ADDR=${KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS#grpc://}
        export TPU_NAME=${KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS}

        notebook_path='/tpu/tools/colab/%(notebook)s'
        jupyter nbconvert --to notebook --inplace --execute "${notebook_path}"

        gsutil cp "${notebook_path}" "$(MODEL_DIR)/%(notebook)s"
      ||| % config.colabSettings
    )
  },

  local fashion_mnist = {
    modelName: 'fashion-mnist',
    colabSettings+:: {
      notebook: 'fashion_mnist.ipynb',
    },
  },
  local iris = {
    modelName: 'iris',
    colabSettings+:: {
      notebook: 'classification_iris_data_with_keras.ipynb',
    },
  },
  local shakespeare = {
    modelName: 'shakespeare',
    colabSettings+:: {
      notebook: 'shakespeare_with_tpu_and_keras.ipynb',
    },
  },

  configs: [
    colab + fashion_mnist,
    colab + iris,
    colab + shakespeare,
  ],
}
