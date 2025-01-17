# Train and test YOLOv8 models
This is the tutorial walking through how to train and test YOLOv8 models.

## System requirements
- Nvidia Drivers
- [Docker Engine](https://docs.docker.com/engine/install/ubuntu/)
- [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

### Model training
- x86
- ubuntu OS
- labeling tools
  - [VGG Image Annotator](https://www.robots.ox.ac.uk/~vgg/software/via/)
  - [Label Studio](https://labelstud.io/)

### TensorRT on GoMax
- arm
- JetPack >= 5.0


## Directory structure
The folder structure below will be created when we go through the tutorial. By convention, we use today's date (i.e. 2023-07-19) as the file name.
```
├── config
│   ├── 2023-07-19_dataset.yaml
│   ├── 2023-07-19_train.yaml
│   ├── 2023-07-19_predict.yaml
│   ├── 2023-07-19_trt.yaml
├── preprocess
│   ├── 2023-07-19.sh
├── data
│   ├── allImages
│   │   ├── *.png
│   │   ├── *.json
├── training
│   ├── 2023-07-19
├── prediction
│   ├── 2023-07-19
├── docker-compose_preprocess.yaml
├── docker-compose_train.yaml
├── docker-compose_predict.yaml
├── docker-compose_trt.yaml
├── dockerfile
├── arm.dockerfile                # arm system
```


## Create a dockerfile
Create a file `./dockerfile`. It installs the dependencies and clone LMI_AI_Solutions repository inside the docker container.
```docker
FROM nvcr.io/nvidia/pytorch:24.04-py3
ARG DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install libgl1 -y
RUN pip install --upgrade pip setuptools wheel
RUN pip install --user opencv-python
RUN pip install ultralytics -U

# clone LMI AI Solutions repository
WORKDIR /repos
RUN git clone https://github.com/lmitechnologies/LMI_AI_Solutions.git

```

## Prepare the dataset
Prepare the dataset by the followings:
- convert json to csv
- resize images and labels in csv
- convert labeling data to YOLO format

**YOLO models require the dimensions of images to be dividable by 32**. In this tutorial, we resize images to 640x640.

### Create a script for data processing
First, create a script `./preprocess/2023-07-19.sh`, which converts labels from label studio json to csv, resizes images, and converts data to yolo format. In the end, it will generate a yolo-formatted dataset in `/app/data/resized_yolo`.
```bash
# modify to your data path
input_path=/app/data/allImages
# modify the width and height according to your data
W=640
H=640

# import the repo paths
source /repos/LMI_AI_Solutions/lmi_ai.env

# convert labels from VGG json to csv
# python -m label_utils.via_json_to_csv -d $input_path --output_fname labels.csv

# convert labels from label studio to csv
python -m label_utils.lst_to_csv -i $input_path -o $input_path

# resize images with labels
python -m label_utils.resize_with_csv -i $input_path -o /app/data/resized --width $W --height $H

# convert to yolo format
# remote the --seg flag if you want to train a object detection model
python -m label_utils.csv_to_yolo -i /app/data/resized -o /app/data/resized_yolo --seg
```

### Create a docker-compose file
To run the bash script in the container, we need to create a file `./docker-compose_preprocess.yaml`.
```yaml
version: "3.9"
services:
  yolov8_preprocess:
    container_name: yolov8_preprocess
    build:
      context: .
      dockerfile: ./dockerfile
    ipc: host
    runtime: nvidia # ensure that Nvidia Container Toolkit is installed
    volumes:
      # mount location_in_host:location_in_container
      - ./data:/app/data
      - ./preprocess/2023-07-19.sh:/app/preprocess/preprocess.sh
    command: >
      bash /app/preprocess/preprocess.sh
```

### Spin up the container
Spin up the container using the following commands: 
```bash
# build the container
# "-f" specifies the yaml file to load
docker compose -f docker-compose_preprocess.yaml build

# spin up the container
# "-f" specifies the yaml file to load
docker compose -f docker-compose_preprocess.yaml up
```
Once it finishs, the yolo format dataset will be created: `./data/resized_yolo`.


## Copy a dataset file indicating the location of the dataset and classes
After converting to yolo format, a dataset yaml file will be created in `./data/resized_yolo/dataset.yaml`. Copy it as `./config/2023-07-19_dataset.yaml`.
```yaml
path: /app/data # dataset root dir (must use absolute path!)
train: images  # train images (relative to 'path')
val: images  # val images (relative to 'path')
test:  # test images (optional)

# Classes
names: # class names must match with the names in class_map.json
  0: peeling
  1: scuff
  2: white
```

If the dataset contains keypoint labels, a `kpt_shape` key will exist in the yaml file. The first number in the `kpt_shape` is the number of keypoints per image, the second number is the number of elements per keypoint. The example below means each image has one keypoint, which has two elements (x,y).
```yaml
kpt_shape:
- 1
- 2
```


## Train the model
To train the model, we need to create a hyperparameter yaml file and create a `./docker-compose_train.yaml` file.

### Create a hyperparameter file
Crete a file `./config/2023-07-19_train.yaml`. Below shows an example of training a **medium-size yolov8 instance segmentation model** with the image size of 640. To train object detection models, set `task` to `detect`. To train keypoint detection models, set it to `pose`.
```yaml
task: segment  # (str) YOLO task, i.e. detect, segment, classify, pose
mode: train  # (str) YOLO mode, i.e. train, predict, export, val, track, benchmark

# training settings
epochs: 300  # (int) number of epochs
batch: 16  # (int) number of images per batch (-1 for AutoBatch)
model: yolov8m-seg.pt # (str) one of yolov8n.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt, yolov8n-seg.pt, yolov8m-seg.pt, yolov8l-seg.pt, yolov8x-seg.pt
imgsz: 640  # (int) input images size, use the larger dimension if rectangular image
patience: 50  # (int) epochs to wait for no observable improvement for early stopping of training
rect: False  # (bool) use rectangular images for training if mode='train' or rectangular validation if mode='val'
exist_ok: False  # (bool) whether to overwrite existing training folder
resume: False  # (bool) resume training from last checkpoint

# data augmentation hyperparameters
degrees: 0.0  # (float) image rotation (+/- deg)
translate: 0.1  # (float) image translation (+/- fraction)
scale: 0.5  # (float) image scale (+/- gain)
shear: 0.0  # (float) image shear (+/- deg)
perspective: 0.0  # (float) image perspective (+/- fraction), range 0-0.001
flipud: 0.0  # (float) image flip up-down (probability)
fliplr: 0.5  # (float) image flip left-right (probability)
mosaic: 1.0  # (float) image mosaic (probability)
mixup: 0.0  # (float) image mixup (probability)
copy_paste: 0.0  # (float) segment copy-paste (probability)

# more hyperparameters: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/default.yaml
```

### Create a docker-compose file
Create a file `./docker-compose_train.yaml`. It mount the host locations to the required directories in the container and run the script `run_cmd.py`, which load the hyperparameters and do the task that was specified in the file `./config/2023-07-19_train.yaml`.
```yaml
version: "3.9"
services:
  yolov8_train:
    container_name: yolov8_train
    build:
      context: .
      dockerfile: dockerfile
    ipc: host
    runtime: nvidia
    ports:
      - 6006:6006 # tensorboard
    volumes:
      - ./training:/app/training   # training output
      - ./data/resized_yolo:/app/data  # training data
      - ./config/2023-07-19_dataset.yaml:/app/config/dataset.yaml  # dataset settings
      - ./config/2023-07-19_train.yaml:/app/config/hyp.yaml  # customized hyperparameters
    command: >
      bash -c "source /repos/LMI_AI_Solutions/lmi_ai.env &&
      python3 -m yolov8_lmi.run_cmd"

```
Note: Do **NOT** modify the required locations in the container, such as `/app/training`, `/app/data`, `/app/config/dataset.yaml`, `/app/config/hyp.yaml`.


### Start training
Spin up the docker containers to train the model as shown in [spin-up-the-container](#spin-up-the-container). **Ensure to load the `docker-compose_train.yaml`.** By default, once the training is done, the run_cmd.py script will create a folder named by today's date in `training` folder, i.e. `training/2023-07-19`.

### Monitor the training progress (optional)
While the training process is running, open another terminal. 
```bash
# shows what containers are currently running
docker ps 

# Log into the container which hosts the training process
# Replace the CONTAINER_ID with actual container ID
docker exec -it CONTAINER_ID bash 

# track the training progress using tensorboard
tensorboard --logdir /app/training/2023-07-19 --port 6006
```

Execuate the command above and go to http://localhost:6006 to monitor the training.


## Prediction
Create a hyperparameter file `./config/2023-07-19_predict.yaml`. The `imgsz` should be a list of [h,w].
```yaml
task: segment  # (str) YOLO task, i.e. detect, segment, classify, pose, where classify, pose are NOT tested
mode: predict  # (str) YOLO mode, i.e. train, predict, export, val, track, benchmark, where track, benchmark are NOT tested

# Prediction settings 
imgsz: 640,640 # (list) input images size as list[h,w] for predict and export modes
conf:  # (float, optional) object confidence threshold for detection (default 0.25 predict, 0.001 val)
max_det: 300  # (int) maximum number of detections per image

# less likely to be used 
show: False  # (bool) show results if possible
save_txt: False  # (bool) save results as .txt file
save_conf: False  # (bool) save results with confidence scores
save_crop: False  # (bool) save cropped images with results
show_labels: True  # (bool) show object labels in plots
show_conf: True  # (bool) show object confidence scores in plots
vid_stride: 1  # (int) video frame-rate stride
line_width:   # (int, optional) line width of the bounding boxes, auto if missing
visualize: False  # (bool) visualize model features
augment: False  # (bool) apply image augmentation to prediction sources
agnostic_nms: False  # (bool) class-agnostic NMS
classes:  # (int | list[int], optional) filter results by class, i.e. classes=0, or classes=[0,2,3]
retina_masks: False  # (bool) use high-resolution segmentation masks
show_boxes: True  # (bool) Show boxes in segmentation predictions

# more hyperparameters: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/default.yaml
```

Create a file `./docker-compose_predict.yaml` as below.
```yaml
version: "3.9"
services:
  yolov8_predict:
    container_name: yolov8_predict
    build:
      context: .
      dockerfile: dockerfile
    ipc: host
    runtime: nvidia
    volumes:
      - ./prediction:/app/prediction  # output path
      - ./training/2023-07-19/weights:/app/trained-inference-models   # trained model path, where it has best.pt
      - ./data/resized_yolo/images:/app/data  # input data path
      - ./config/2023-07-19_predict.yaml:/app/config/hyp.yaml  # customized hyperparameters
    command: >
      bash -c "source /repos/LMI_AI_Solutions/lmi_ai.env &&
      python3 -m yolov8_lmi.run_cmd"

```

### Start prediction
Spin up the container as shown in [spin-up-the-container](#spin-up-the-container). **Ensure to load the `docker-compose_predict.yaml.`** Then, the output results are saved in `./prediction/2023-07-19`.


## Generate TensorRT engines
The TensorRT egnines can be generated in two systems: x86 and arm. Both systems share the same hyperparameter file, while the dockerfile and docker-compose files are different.

Create a hyperparamter yaml file `./config/2023-07-19_trt.yaml`:
```yaml
task: segment  # (str) YOLO task, i.e. detect, segment, classify, pose, where classify, pose are NOT tested
mode: export  # (str) YOLO mode, i.e. train, predict, export, val, track, benchmark, where track, benchmark are NOT tested

# Export settings 
format: engine  # (str) format to export to, choices at https://docs.ultralytics.com/modes/export/#export-formats
half: True  # (bool) use half precision (FP16)
imgsz: 640,640 # (list) input images size as list[h,w] for predict and export modes
device: 0 # (int | str | list, optional) device to run on, i.e. cuda device=0 or device=0,1,2,3 or device=cpu

# less likely used settings 
dynamic: False  # (bool) ONNX/TF/TensorRT: dynamic axes
simplify: False  # (bool) ONNX: simplify model
opset:  # (int, optional) ONNX: opset version
workspace: 4  # (int) TensorRT: workspace size (GB)

# more hyperparameters: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/default.yaml
```

Create a docker-compose file `./docker-compose_trt.yaml`:
```yaml
version: "3.9"
services:
  yolov8_trt:
    container_name: yolov8_trt
    build:
      context: .
      dockerfile: dockerfile
    ipc: host
    runtime: nvidia
    volumes:
      - ./training/2023-07-19/weights:/app/trained-inference-models   # trained model path, which includes a best.pt
      - ./config/2023-07-19_trt.yaml:/app/config/hyp.yaml  # customized hyperparameters
    command: >
      bash -c "source /repos/LMI_AI_Solutions/lmi_ai.env &&
      python3 -m yolov8_lmi.run_cmd"
```

### Engine Generation on x86 systems
#### Start generation
Spin up the container as shown in [spin-up-the-container](#spin-up-the-container). **Ensure to load the `docker-compose_trt.yaml`.** Then, the tensorRT engine is generated in `./training/2023-07-19/weights`.


### Engine Generation on arm systems
Create a file `./arm.dockerfile`.
```docker
# jetpack 5.1
FROM --platform=linux/arm64/v8 nvcr.io/nvidia/l4t-ml:r35.2.1-py3
ARG DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN python3 -m pip install pip --upgrade
RUN pip3 install --upgrade setuptools wheel
RUN pip3 install opencv-python --user
RUN pip3 install ultralytics -U

# clone AIS
WORKDIR /repos
RUN git clone https://github.com/lmitechnologies/LMI_AI_Solutions.git
```

Replace the line `dockerfile: dockerfile` in `./docker-compose_trt.yaml` with `dockerfile: arm.dockerfile`.

#### Start generation
Spin up the container as shown in [spin-up-the-container](#spin-up-the-container). Ensure to load the `./docker-compose_trt.arm.yaml`. The output engines are saved in `./training/2023-07-19/weights`.