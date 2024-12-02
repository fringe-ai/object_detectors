from typing import Dict, List
from model_base import ModelBase
import tensorrt as trt
from cuda import cudart
import numpy as np
import detectron2_lmi.utils.common_runtime as common
from gadget_utils.pipeline_utils import plot_one_box, revert_to_origin
from postprocess_utils.mask_utils import rescale_masks,mask_to_polygon
import cv2
import logging
import torch

import time


class Detectron2ModelFactory:
    @staticmethod
    def load_model(model_path, class_map):
        if model_path.endswith(".ts"):
            return Detectron2Torchscript(model_path, class_map)
        elif model_path.endswith(".engine"):
            return Detectron2TRT(model_path, class_map)
        else:
            raise NotImplementedError(f"Model type not supported: {model_path}")
    

class Detectron2TRT(ModelBase):
    logger = logging.getLogger(__name__)
    def __init__(self, model_path, class_map):
        """
        Initialize the Detectron2 model with TensorRT engine.
        Args:
            model_path (str): Path to the serialized TensorRT engine file.
            class_map (dict): Dictionary mapping class IDs to class names.
        Attributes:
            engine (trt.ICudaEngine): The TensorRT engine.
            context (trt.IExecutionContext): The execution context for the engine.
            model_inputs (list): List of input tensor bindings.
            model_outputs (list): List of output tensor bindings.
            allocations (list): List of memory allocations for input and output tensors.
            input_shape (list): Shape of the input tensor.
            input_dtype (numpy.dtype): Data type of the input tensor.
            class_map (dict): Dictionary mapping class IDs to class names.
        """
        """source: https://github.com/NVIDIA/TensorRT/tree/release/10.4/samples/python/detectron2"""
        
        trt_logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(trt_logger, namespace="")
        with open(model_path, "rb") as f, trt.Runtime(trt_logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        # Setup I/O bindings
        self.model_inputs = []
        self.model_outputs = []
        self.allocations = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            is_input = False
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                is_input = True
            dtype = self.engine.get_tensor_dtype(name)
            shape = self.engine.get_tensor_shape(name)
            if is_input:
                self.batch_size = shape[0]
            size = np.dtype(trt.nptype(dtype)).itemsize
            for s in shape:
                size *= s
            allocation = common.cuda_call(cudart.cudaMalloc(size))
            binding = {
                "index": i,
                "name": name,
                "dtype": np.dtype(trt.nptype(dtype)),
                "shape": list(shape),
                "allocation": allocation,
                "size": size,
            }
            self.allocations.append(allocation)
            if is_input:
                self.model_inputs.append(binding)
            else:
                self.model_outputs.append(binding)
        
        self.input_shape = self.model_inputs[0]["shape"]
        self.input_dtype = self.model_inputs[0]["dtype"]

        self.class_map = {
            int(k): str(v) for k, v in class_map.items()
        }
        self.class_map_func = np.vectorize(lambda c: self.class_map.get(int(c), str(c)))
    
    def warmup(self):
        """
        Perform a warmup operation for the model.

        This method runs a forward pass with a randomly generated input tensor
        to warm up the model. It helps in preparing the model for actual inference
        by initializing necessary components and reducing the initial latency.

        The input tensor is generated with the same shape and data type as the
        expected input during inference.

        Parameters:
        None

        Returns:
        None
        """
        for _ in range(1):
            image_h, image_w = self.input_shape[2], self.input_shape[3]
            input = np.random.rand(self.batch_size, 3, image_h, image_w).astype(self.input_dtype)
            self.forward(input)
        
    def preprocess(self, images: np.ndarray):
        """
        Preprocesses a batch of images for input into the model.

        Args:
            images (np.ndarray): A batch of images to preprocess. Each image should be in the format (H, W, C).

        Returns:
            np.ndarray: A batch of preprocessed images with shape (batch_size, 3, image_h, image_w).
        """
        image_h, image_w = self.input_shape[2], self.input_shape[3]
        inputs = np.zeros((self.batch_size, 3, image_h, image_w), dtype=self.input_dtype)
        for i in range(0, self.batch_size):
            image = images[i]
            inputs[i] = image.astype(np.float32).transpose(2, 0, 1)
        return inputs
    
    def forward(self, inputs):
        """
        Perform a forward pass through the model.

        Args:
            inputs (numpy.ndarray): The input data to be processed by the model.

        Returns:
            list: A list of numpy arrays containing the model's output data.
        """
        outputs = []
        for out in self.model_outputs:
            outputs.append(np.zeros(out["shape"], dtype=out["dtype"]))
        common.memcpy_host_to_device(
            self.model_inputs[0]["allocation"], np.ascontiguousarray(inputs)
        )
        self.context.execute_v2(self.allocations)
        for o in range(len(outputs)):
            common.memcpy_device_to_host(outputs[o], self.model_outputs[o]["allocation"])
        return outputs
    
    def postprocess(self, images, predictions,operators=[], **kwargs):
        """
        Post-process the predictions from the object detection model.
        Args:
            images (list): List of input images.
            predictions (tuple): Tuple containing the number of predictions, bounding boxes, scores, classes, and masks.
            **kwargs: Additional keyword arguments for processing.
                - confs (dict): Dictionary of confidence thresholds for each class.
                - mask_threshold (float): Threshold for mask binarization.
                - process_masks (bool): Flag to indicate whether to process masks.
        Returns:
            dict: A dictionary containing the processed results with keys:
                - "boxes" (list): List of processed bounding boxes for each image.
                - "scores" (list): List of processed scores for each image.
                - "classes" (list): List of processed class labels for each image.
                - "masks" (list): List of processed masks for each image.
        """
        results = {
            "boxes": [],
            "scores": [],
            "classes": [],
            "masks": [],
            "segments": []
        }
        
        if len(predictions) == 0:
            return results
        confs = kwargs.get("confs", {})
        mask_threshold = kwargs.get("mask_threshold", 0.5)
        process_masks = kwargs.get("process_masks", False)
        image_h, image_w = images[0].shape[0], images[0].shape[1]
        
        
        num_preds, boxes, scores, classes, masks = predictions[:5]

        
        classes = self.class_map_func(classes)
        
        if len(boxes) > 0:
            scale_factors = np.array([image_w, image_h, image_w, image_h])
            boxes = (boxes * scale_factors).astype(np.int32)
            
        processed_masks = []
        processed_boxes = []
        processed_scores = []
        processed_classes = []
        processed_segments = []


        t0 = time.time()
        for idx in range(self.batch_size):
            valid_scores = scores[idx] >= np.vectorize(confs.get)(classes[idx], 1.0)
            batch_boxes, batch_scores = boxes[idx][valid_scores], scores[idx][valid_scores]
            batch_classes = classes[idx][valid_scores]
            batch_segments = []
            processed_boxes.append(batch_boxes)
            processed_scores.append(batch_scores)
            processed_classes.append(batch_classes)
            filtered_masks = masks[idx][valid_scores]
            batch_masks = []
            if process_masks:
                batch_masks = rescale_masks(
                    torch.from_numpy(filtered_masks).to(self.device),
                    torch.from_numpy(batch_boxes).to(self.device),
                    (image_h, image_w,),
                    mask_threshold
                )
                if kwargs.get("return_segments", False):
                    if len(operators) > 0:
                        batch_segments = [
                            revert_to_origin(mask_to_polygon(
                                mask
                                ), operators) for mask in batch_masks
                        ]
                    else:
                        batch_segments = [
                            mask_to_polygon(mask) for mask in batch_masks
                        ]
                    
            else:
                batch_masks = filtered_masks
            
            # apply revert to origin
            batch_boxes = revert_to_origin(batch_boxes, operators)
            processed_masks.append(batch_masks)
            processed_segments.append(batch_segments)
            
        t1 = time.time()
        self.logger.info(f"post-processing time {(t1-t0)* 1000.0} ms")     
        results = {
            "boxes": processed_boxes,
            "scores": processed_scores,
            "classes": processed_classes,
            "masks": processed_masks,
            "segments": processed_segments
        }
        return results
    
    def get_batches(self, images):
        # get the number of batches add empty images to make it a multiple of batch size
        num_batches = len(images) // self.batch_size
        if len(images) % self.batch_size != 0:
            num_batches += 1
            for i in range(0, self.batch_size - len(images) % self.batch_size):
                images.append(np.zeros_like(images[0]))
        for i in range(0, num_batches):
            yield images[i * self.batch_size : (i + 1) * self.batch_size]
    

    def predict(self, images, **kwargs):
        """
        Perform prediction on the given images.

        Args:
            images (list or np.ndarray): The input images to be processed.
            **kwargs: Additional keyword arguments for postprocessing.

        Returns:
            list: The predictions after postprocessing.

        Logs:
            The time taken for postprocessing in milliseconds.
        """
        predictions = self.forward(self.preprocess(images))
        t0 = time.time()
        predictions = self.postprocess(images, predictions,**kwargs)
        t1 = time.time()
        self.logger.info(f"postprocessing time {(t1-t0)*1000.0:.2f} ms")
        return predictions
    
    def annotate_image(self, result, image, color_map=None):
        """
        Annotates an image with bounding boxes, class labels, and masks.

        Args:
            result (dict): A dictionary containing detection results with keys:
                - "classes" (list): List of class labels for detected objects.
                - "boxes" (list): List of bounding boxes for detected objects.
                - "scores" (list): List of confidence scores for detected objects.
                - "masks" (list, optional): List of masks for detected objects.
            image (numpy.ndarray): The image to annotate.
            color_map (dict, optional): A dictionary mapping class labels to colors.

        Returns:
            numpy.ndarray: The annotated image.
        """
        for i in range(len(result["classes"])):
            plot_one_box(
                result["boxes"][i],
                image,
                label=f"{result['classes'][i]}:{result['scores'][i]:.2f}",
                mask=result["masks"][i] if len(result["masks"]) > 0 else None,
                color=color_map,
            )
        return image

class Detectron2Torchscript(ModelBase):
    logger = logging.getLogger(__name__)
   
    def __init__(self, model_path, class_map, device='cuda', **kwargs):
        try:
            self.model = torch.jit.load(model_path)
        except Exception as e:
            self.logger.exception(f"Failed to load model: {e}")
        
        self.device = torch.device(device)
        # move the model to gpu
        self.model.to(self.device)
        
        self.class_map = {
            int(k): str(v) for k, v in class_map.items()
        }
        self.batch_size = kwargs.get('batch_size', 1)
        self.class_map_func = np.vectorize(lambda c: self.class_map.get(int(c), str(c)))
    
    def warmup(self, **kwargs):
        """
        Perform a warmup operation for the model.

        This method generates a random input tensor with the specified image size and 
        passes it through the model to perform a warmup. This can be useful to initialize 
        model parameters and optimize performance before actual inference.

        Args:
            **kwargs: Arbitrary keyword arguments.
                img_size (tuple, optional): A tuple specifying the height and width of the image.
                                            If not provided, an error is logged.

        Raises:
            ValueError: If 'img_size' is not provided in kwargs.
        """
        image_size = kwargs.get('img_size', [1024,1024])
        image_h, image_w = image_size[0], image_size[1]
        images = [np.random.rand(image_h, image_w, 3).astype(np.float32) for _ in range(self.batch_size)]
        input = [
            dict(image=torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1).to(dtype=torch.float32).to(self.device)) for image in images
        ]
        self.forward(input)
        
    def preprocess(self, images: np.ndarray) -> List[Dict[str, torch.Tensor]]:
        """
        Preprocesses a batch of images for input into the model.

        Args:
            images (np.ndarray): A numpy array of images to be preprocessed. 
                                 Each image is expected to be in HWC format.

        Returns:
            list: A list of dictionaries where each dictionary contains a single key 'image' 
                  with the preprocessed image as a value. The image is converted to float32 
                  and transposed to CHW format.
        """
        image_h, image_w = images[0].shape[0], images[0].shape[1]
        inputs = np.zeros((len(images), 3, image_h, image_w), dtype=np.float32)
        inputs = [
            dict(image=torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1).to(dtype=torch.float32).to(self.device)) for image in images
        ]
        return inputs
    
    def forward(self, inputs):
        """
        Perform a forward pass through the model with the given inputs.

        Args:
            inputs (torch.Tensor): The input tensor to be passed through the model.

        Returns:
            torch.Tensor: The model's predictions for the given inputs.
        """
        with torch.no_grad():
            predictions = self.model(inputs)
        return predictions
    
    def postprocess(self, images, predictions,operators=[], **kwargs):
        """
        Post-processes the predictions from the model to filter and format the results.
        Args:
            images (list): A list of input images.
            predictions (list): A list of predictions from the model, where each prediction is a dictionary containing:
            - "scores" (Tensor): Confidence scores for each detected object.
            - "pred_classes" (Tensor): Predicted class indices for each detected object.
            - "pred_boxes" (Tensor): Bounding boxes for each detected object.
            - "pred_masks" (Tensor, optional): Segmentation masks for each detected object.
            **kwargs: Additional keyword arguments for post-processing:
            - confs (dict): A dictionary mapping class indices to confidence thresholds.
            - mask_threshold (float): Threshold for binarizing masks.
        Returns:
            dict: A dictionary containing the post-processed results with the following keys:
            - "boxes" (list): A list of numpy arrays containing bounding boxes for each image.
            - "scores" (list): A list of numpy arrays containing confidence scores for each image.
            - "classes" (list): A list of numpy arrays containing class indices for each image.
            - "masks" (list): A list of numpy arrays containing segmentation masks for each image (if available).
        """
        results = {
            "boxes": [],
            "scores": [],
            "classes": [],
            "masks": [],
            "segments": []
        }
        
        if len(predictions) == 0:
            return results
        confs = kwargs.get("confs", {})
        mask_threshold = kwargs.get("mask_threshold", 0.5)
        
        # if no predictions, return empty results
        if predictions[0]["pred_classes"].shape[0] == 0:
            return results
        for idx, output in enumerate(predictions):
            image_h, image_w = images[idx].shape[:2]
            batch_scores = output["scores"].cpu().numpy()
            batch_classes = output["pred_classes"].cpu().numpy()
            batch_segments = []
            batch_masks = []
            batch_classes = self.class_map_func(batch_classes)
            keep = batch_scores >= np.vectorize(confs.get)(batch_classes, 0.5)
            batch_scores = batch_scores[keep]
            batch_classes = batch_classes[keep]
            batch_boxes = output["pred_boxes"][keep]
            if "pred_masks" in output:
                batch_masks = output["pred_masks"][keep]
                batch_masks = rescale_masks(batch_masks.to(self.device).squeeze(1), batch_boxes.to(self.device), (image_h,image_w,), mask_threshold)
                if kwargs.get("return_segments", False):
                    if len(operators) > 0:
                        batch_segments = [
                            revert_to_origin(mask_to_polygon(
                                mask
                                ), operators) for mask in batch_masks
                        ]
                    else:
                        batch_segments = [
                            mask_to_polygon(mask) for mask in batch_masks
                        ]
            
            batch_boxes = revert_to_origin(batch_boxes, operators)
            results["boxes"].append(batch_boxes.cpu().numpy())
            results["scores"].append(batch_scores)
            results["classes"].append(batch_classes)
            results["masks"].append(batch_masks.cpu().numpy() if isinstance(batch_masks, torch.Tensor) else batch_masks)
            results["segments"].append(batch_segments)
        return results
    
    def predict(self, images, operators=[],**kwargs):
        """
        Perform prediction on the given images.

        This method preprocesses the input images, performs forward pass to get predictions,
        and then postprocesses the predictions to generate the final results.

        Args:
            images (list or array-like): The input images to be processed.
            operators (optional): Not yet supported. Default is None.
            **kwargs: Additional keyword arguments for postprocessing.

        Returns:
            results: The final processed results after prediction.

        Raises:
            NotImplementedError: If operators is not None, indicating that the feature is not yet supported.
        """
        # preprocess
        inputs = self.preprocess(images)
        # forward
        predictions = self.forward(inputs)
        # postprocess
        results = self.postprocess(images, predictions, operators=operators,**kwargs)
        return results
        
    
    def annotate_image(self, result, image, color_map=None):
        """
        Annotates an image with bounding boxes, class labels, scores, and masks.

        Args:
            result (dict): A dictionary containing detection results with keys:
                - "classes" (list): List of detected class labels.
                - "scores" (list): List of confidence scores for each detected class.
                - "boxes" (list): List of bounding boxes for each detected object.
                - "masks" (list, optional): List of masks for each detected object.
            image (numpy.ndarray): The image to annotate.
            color_map (dict, optional): A dictionary mapping class labels to colors.

        Returns:
            numpy.ndarray: The annotated image.
        """
        for i in range(len(result["classes"])):
            plot_one_box(
                result["boxes"][i],
                image,
                label=f"{result['classes'][i]}:{result['scores'][i]:.2f}",
                mask=result["masks"][i] if len(result["masks"]) > 0 else None,
                color=color_map,
            )
        return image


if __name__ == "__main__":
    import argparse
    import glob
    import os
    import time
    import json
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/home/weights/model.engine")
    parser.add_argument("--images_path", type=str, default="/home/input")
    parser.add_argument("--class-map", type=str, help="Class map json file", default="/home/class_map.json")

    
    args = parser.parse_args()
    with open(args.class_map, "r") as f:
        class_map = json.load(f)
    model = Detectron2TRT(args.model_path, class_map)
    model.warmup()
    
    images = glob.glob(os.path.join(args.images_path, "*"))
    image_batch = [
        cv2.imread(image)
        for image in images
    ]
    batches = model.get_batches(image_batch)
    current_image_count = 0
    for idx, batch in enumerate(batches):
        batch_preds = model.predict(batch)
        num_images = len(batch_preds["classes"])
        batch_classes = batch_preds["classes"]
        # remove empty images
        for i in range(num_images):
            if i + current_image_count >= len(images):
                break
            image_path = images[i + current_image_count]
            image = cv2.imread(image_path)
            batch_pred = {
                "boxes": batch_preds["boxes"][i],
                "scores": batch_preds["scores"][i],
                "classes": batch_preds["classes"][i],
            }
            annotated_image = model.annotate_image(batch_pred, image)
            file_ext = os.path.basename(image_path).split('.')[-1]
            cv2.imwrite(f"/home/data/output/{os.path.basename(image_path)}_annotated.{file_ext}", annotated_image)
        current_image_count += num_images
