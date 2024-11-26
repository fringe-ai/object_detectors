## Training MaskRCNN

### Dataset

The required dataset format is COCO, in the following structure:


```text
<DATASETS DIR>/
    -<DATASET_NAME>/
        --images/
            ...
        --annotations.json
```
example:

```text
coco/
    train/
        images/
            ...
        annotations.json
```

*Dataset name should be the same name as the name declared in the config yaml file*

### Training

- [Configuration](#configuration)
- [Dockerfile](#dockerfile)
- [Training](#train)

#### Configuration

The following is an example configuration file for training a maskrcnn model

```yaml
DATASETS:
  TEST: [] # define the test dataset similar to the train dataset or leave it as empty list (use the name "test_dataset")
  TRAIN:
  - train # dataset name should make the folder name

MODEL:
  ROI_HEADS:
    NUM_CLASSES: 1
  
INPUT:
  MIN_SIZE_TRAIN: 
  - 512
  MAX_SIZE_TRAIN: 512
  MIN_SIZE_TEST: 512
  MAX_SIZE_TEST: 512

# Training parameters
SOLVER:
  AMP:
    ENABLED: false
  BASE_LR: 0.02
  BIAS_LR_FACTOR: 1.0
  CHECKPOINT_PERIOD: 5000 # Save a checkpoint after every N number of iterations
  GAMMA: 0.1
  IMS_PER_BATCH: 2 # Number of images per batch
  LR_SCHEDULER_NAME: WarmupMultiStepLR
  MAX_ITER: 6000 # Maximum number of iterations default is 270000 for 3x schedule please change it accordingly

TEST:
  # Feel free to add more augmentations or change the min/max sizes for the images
  DETECTIONS_PER_IMAGE: 1000 # Number of detections per image
  EVAL_PERIOD: 0 # Runs evaluation every N iterations
```

#### Dockerfile

##### x86
```dockerfile
FROM nvcr.io/nvidia/pytorch:23.04-py3
ARG DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install libgl1 -y
RUN pip install --upgrade pip setuptools wheel
RUN pip install --user opencv-python
RUN pip install ultralytics -U

# clone LMI AI Solutions repository
WORKDIR /home
RUN python -m pip install --upgrade pip
RUN pip install torch torchvision torchaudio
RUN pip install --user 'git+https://github.com/facebookresearch/fvcore'
RUN git clone https://github.com/facebookresearch/detectron2 detectron2
RUN pip install --user -e detectron2 
RUN pip install tensorboard
RUN git clone -b FAIE-1767-v0.0.6 https://github.com/lmitechnologies/LMI_AI_Solutions.git
RUN pip install onnx-graphsurgeon onnxruntime
```

#### Train

Example docker-compose.yaml file to start a training job

```yaml
version: "3.9"
services:
  detectron2_lmi:
    container_name: detectron2_lmi
    build:
      context: .
      dockerfile: dockerfile
    ipc: host
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    ports:
      - 6006:6006 # tensorboard
    volumes:
      - ./training/maskrcnn/:/home/weights   # training output where trained models are saved
      - ./data/coco/:/home/data  # training data please attach to the root folder
      - ./configs/maskrcnn.yaml:/home/config.yaml  # customized hyperparameters
    command: >
      python3 /home/LMI_AI_Solutions/object_detectors/detectron2_lmi/trainer.py
```
*The training process automatically starts tensorboard*

##### Tensorboard

Served up at the following address [localhost:6006](http://localhost:6006)
*6006 is the default port*


#### Inference

To run inference please use the following docker-compose.yaml file:

```yaml
version: "3.9"
services:
  detectron2_lmi:
    container_name: detectron2_lmi
    build:
      context: .
      dockerfile: dockerfile
    ipc: host
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    ports:
      - 6006:6006 # tensorboard
    volumes:
      - ./training/maskrcnn/2024-09-22-v2/:/home/weights   # trained model weights
      - ./data/coco/train/images/:/home/input  # folder of images
      - ./prediction/maskrcnn/2024-09-22-v2-1/:/home/output  # annotated images output
      - ./configs/maskrcnn.yaml:/home/config.yaml  # customized hyperparameters for test can be the same as train
      - ./configs/class_map.json:/home/class_map.json  # class map file int:string
    command: >
      bash -c "source /home/LMI_AI_Solutions/lmi_ai.env && python3 /home/LMI_AI_Solutions/object_detectors/detectron2_lmi/inference.py --confidence-threshold 0.1"
```
###### Outputs:

A LMI formated csv file with all predictions is automatically saved in the the the output folder defined in the docker compose file.

### Convert to TensorRT

To convert to tensorrt a `sample_image.png` is required to be in folder where the weights are stored. The image should be of size thats divizeable by 32. The imagesize should be defined in the config.yaml file shown above for training.

*Default batch size is 1 although batchsize can be changed to any batchsize*

```yaml
services:
  detectron2_lmi:
    container_name: detectron2_lmi
    build:
      context: .
      dockerfile: dockerfile
    ipc: host
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    ports:
      - 6006:6006 # tensorboard
    volumes:
      - ./training/maskrcnn/2024-09-22-v2/:/home/weights   # weights
    command: >
      bash -c "source /home/LMI_AI_Solutions/lmi_ai.env && python3 /home/LMI_AI_Solutions/object_detectors/detectron2_lmi/convert.py -b 1 --fp16"
```