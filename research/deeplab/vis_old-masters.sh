
# Exit immediately if a command exits with a non-zero status.
set -e

#export CUDA_VISIBLE_DEVICES="0"

# Move one-level up to tensorflow/models/research directory.
cd ..

# Update PYTHONPATH.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}/deeplab"
DATASET_DIR="datasets"
MINC_SEGMENTATION_FOLDER="minc-segmentation"
EXP_FOLDER="exp/default_train_finetune"
INIT_FOLDER="${WORK_DIR}/${DATASET_DIR}/${MINC_SEGMENTATION_FOLDER}/init_models"
TRAIN_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${MINC_SEGMENTATION_FOLDER}/${EXP_FOLDER}/train"
EVAL_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${MINC_SEGMENTATION_FOLDER}/${EXP_FOLDER}/eval"
#VIS_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${MINC_SEGMENTATION_FOLDER}/${EXP_FOLDER}/vis"
VIS_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${MINC_SEGMENTATION_FOLDER}/${EXP_FOLDER}/vis-old-masters"
EXPORT_DIR="${WORK_DIR}/${DATASET_DIR}/${MINC_SEGMENTATION_FOLDER}/${EXP_FOLDER}/export"
#MINC_SEGMENTATION_DATASET="${WORK_DIR}/${DATASET_DIR}/${MINC_SEGMENTATION_FOLDER}/tfrecord"
MINC_SEGMENTATION_DATASET="${WORK_DIR}/${DATASET_DIR}/old-masters/tfrecord"
mkdir -p "${INIT_FOLDER}"
mkdir -p "${TRAIN_LOGDIR}"
mkdir -p "${EVAL_LOGDIR}"
mkdir -p "${VIS_LOGDIR}"
mkdir -p "${EXPORT_DIR}"

# Visualize the results.
python "${WORK_DIR}"/vis.py \
  --logtostderr \
  --vis_split="test" \
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
  --dataset=old-masters \ #--dataset=minc-segmentation \
  --colormap_type=minc-segmentation \
  --max_number_of_iterations=1
