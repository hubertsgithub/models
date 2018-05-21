#!/bin/bash
# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# This script is used to run local test on PASCAL VOC 2012. Users could also
# modify from this script for their use case.
#
# Usage:
#   # From the tensorflow/models/research/deeplab directory.
#   sh ./local_test.sh
#
#

# Exit immediately if a command exits with a non-zero status.
set -e

export CUDA_VISIBLE_DEVICES="0"

# Move one-level up to tensorflow/models/research directory.
cd ..

# Update PYTHONPATH.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

# Set up the working environment.
CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}/deeplab"

# Run model_test first to make sure the PYTHONPATH is correctly set.
#python "${WORK_DIR}"/model_test.py -v

DATASET_DIR="datasets"

# Go back to original directory.
cd "${CURRENT_DIR}"

# Set up the working directories.
MINC_SEGMENTATION_FOLDER="minc-segmentation"
#EXP_FOLDER="exp/default_train_finetune"
EXP_FOLDER="exp/test_hint_train_finetune"
INIT_FOLDER="${WORK_DIR}/${DATASET_DIR}/${MINC_SEGMENTATION_FOLDER}/init_models"
TRAIN_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${MINC_SEGMENTATION_FOLDER}/${EXP_FOLDER}/train"
TRAIN_CHKPT_BACKUP_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${MINC_SEGMENTATION_FOLDER}/${EXP_FOLDER}/train/backup_checkpoints"
EVAL_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${MINC_SEGMENTATION_FOLDER}/${EXP_FOLDER}/eval"
VIS_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${MINC_SEGMENTATION_FOLDER}/${EXP_FOLDER}/vis"
EXPORT_DIR="${WORK_DIR}/${DATASET_DIR}/${MINC_SEGMENTATION_FOLDER}/${EXP_FOLDER}/export"
mkdir -p "${INIT_FOLDER}"
mkdir -p "${TRAIN_LOGDIR}"
mkdir -p "${TRAIN_CHKPT_BACKUP_LOGDIR}"
mkdir -p "${EVAL_LOGDIR}"
mkdir -p "${VIS_LOGDIR}"
mkdir -p "${EXPORT_DIR}"

## Copy locally the trained checkpoint as the initial checkpoint.
#TF_INIT_ROOT="http://download.tensorflow.org/models"
#TF_INIT_CKPT="deeplabv3_pascal_train_aug_2018_01_04.tar.gz"
#cd "${INIT_FOLDER}"
#wget -nd -c "${TF_INIT_ROOT}/${TF_INIT_CKPT}"
#tar -xf "${TF_INIT_CKPT}"
cd "${CURRENT_DIR}"

MINC_SEGMENTATION_DATASET="${WORK_DIR}/${DATASET_DIR}/${MINC_SEGMENTATION_FOLDER}/tfrecord"

# Train 10 iterations.
#NUM_ITERATIONS=10000
NUM_ITERATIONS=$@
python "${WORK_DIR}"/train.py \
  --logtostderr \
  --train_split="train_class_hints" \
  --model_variant="xception_65" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --train_crop_size=513 \
  --train_crop_size=513 \
  --resize_factor=16 \
  --train_batch_size=4 \
  --training_number_of_steps="${NUM_ITERATIONS}" \
  --fine_tune_batch_norm=False \
  --initialize_last_layer=False \
  --last_layers_contain_logits_only=True \
  --tf_initial_checkpoint="${INIT_FOLDER}/deeplabv3_pascal_train_aug/model.ckpt" \
  --train_logdir="${TRAIN_LOGDIR}" \
  --dataset_dir="${MINC_SEGMENTATION_DATASET}" \
  --dataset=minc-segmentation \
  --class_hints=True

cp -v ${TRAIN_LOGDIR}/model.ckpt-${NUM_ITERATIONS}* ${TRAIN_CHKPT_BACKUP_LOGDIR}

# Run evaluation. This performs eval over the full val split.
python "${WORK_DIR}"/eval.py \
  --logtostderr \
  --eval_split="val_class_hints" \
  --model_variant="xception_65" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --eval_crop_size=513 \
  --eval_crop_size=513 \
  --resize_factor=16 \
  --checkpoint_dir="${TRAIN_LOGDIR}" \
  --eval_logdir="${EVAL_LOGDIR}" \
  --dataset_dir="${MINC_SEGMENTATION_DATASET}" \
  --dataset=minc-segmentation \
  --max_number_of_evaluations=1 \
  --class_hints=True

# Visualize the results.
#VIS_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${MINC_SEGMENTATION_FOLDER}/${EXP_FOLDER}/vis"
VIS_LOGDIR="${VIS_LOGDIR}/${NUM_ITERATIONS}"
mkdir -p "${VIS_LOGDIR}"
python "${WORK_DIR}"/vis.py \
  --logtostderr \
  --vis_split="val_class_hints" \
  --model_variant="xception_65" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --vis_crop_size=513 \
  --vis_crop_size=513 \
  --resize_factor=16 \
  --checkpoint_dir="${TRAIN_LOGDIR}" \
  --vis_logdir="${VIS_LOGDIR}" \
  --dataset_dir="${MINC_SEGMENTATION_DATASET}" \
  --dataset=minc-segmentation \
  --class_hints=True \
  --max_number_of_iterations=1 \
  --colormap_type=minc-segmentation

## Export the trained checkpoint.
#CKPT_PATH="${TRAIN_LOGDIR}/model.ckpt-${NUM_ITERATIONS}"
#EXPORT_PATH="${EXPORT_DIR}/frozen_inference_graph.pb-${NUM_ITERATIONS}"
#
#python "${WORK_DIR}"/export_model.py \
#  --logtostderr \
#  --checkpoint_path="${CKPT_PATH}" \
#  --export_path="${EXPORT_PATH}" \
#  --model_variant="xception_65" \
#  --atrous_rates=6 \
#  --atrous_rates=12 \
#  --atrous_rates=18 \
#  --output_stride=16 \
#  --decoder_output_stride=4 \
#  --num_classes=23 \
#  --crop_size=513 \
#  --crop_size=513 \
#  --inference_scales=1.0 \
#  --class_hints=True

# Run inference with the exported checkpoint.
# Please refer to the provided deeplab_demo.ipynb for an example.
