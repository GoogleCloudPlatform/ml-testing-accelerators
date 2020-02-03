local bert_mnli = import "bert-mnli.libsonnet";
local bert_squad = import "bert-squad.libsonnet";
local mnist = import "mnist.libsonnet";
local resnet_ctl = import "resnet-ctl.libsonnet";
local resnet_cfit = import "resnet-cfit.libsonnet";
local retinanet = import "retinanet.libsonnet";
local transformer_translate = import "transformer-translate.libsonnet";

# Add new models here
std.flattenArrays([
  bert_mnli.configs,
  bert_squad.configs,
  mnist.configs,
  resnet_ctl.configs,
  resnet_cfit.configs,
  retinanet.configs,
  transformer_translate.configs,
])
