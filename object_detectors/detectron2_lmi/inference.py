from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import detectron2.data.transforms as T
from detectron2.modeling import build_model
import torch
from detectron2.checkpoint import DetectionCheckpointer
from ..model_base import ModelBase

class Detectron2Model(ModelBase):
    
    def __init__(self,weights_path: str, config_file: str):
        """
        The function initializes a model with specified weights and configuration for object detection tasks
        in Python.
        
        :param weights_path: The `weights_path` parameter is the file path to the pre-trained weights of the
        model that you want to load for inference. This file typically contains the learned parameters of
        the model after training on a specific dataset
        :type weights_path: str
        :param config_file: The `config_file` parameter is a string that represents the file path to a
        configuration file. This configuration file is used to configure the model and its components, such
        as the model architecture, input sizes, and other settings required for the model to make
        predictions
        :type config_file: str
        """
        
        self.cfg = get_cfg()
        self.cfg.merge_from_file(config_file)
        self.model = build_model(self.cfg)
        self.model.eval()
        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(weights_path)
        self.model = DefaultPredictor(self.cfg).model
        self.transforms = T.ResizeShortestEdge([self.cfg.INPUT.MIN_SIZE_TEST, self.cfg.INPUT.MIN_SIZE_TEST], self.cfg.INPUT.MAX_SIZE_TEST)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def warmup(self):
        """
        The `warmup` function initializes a tensor input and runs the model on it for 10 iterations without
        gradient computation.
        """
        input= torch.tensor([1,3,self.cfg.INPUT.MAX_SIZE_TEST,self.cfg.INPUT.MAX_SIZE_TEST])
        with torch.no_grad():
            for i in range(10):
                self.model(input)

    def preprocess(self, image):
        """
        The `preprocess` function takes an image, applies transformations, converts it to a tensor, and
        returns the processed image along with its height and width.
        
        :param image: The `image` parameter is a NumPy array representing an image. The `preprocess` method
        takes this image as input and performs some preprocessing steps on it before returning a dictionary
        containing the preprocessed image as a PyTorch tensor, along with the height and width of the
        original image
        :return: The preprocess method returns a dictionary with the following keys and values:
        - "image": torch tensor of the preprocessed input image, converted to float32 and transposed to (C,
        H, W) format, then moved to the device specified in the class
        - "height": the height of the original image
        - "width": the width of the original image
        """
        height, width = image.shape[:2]
        input = self.transforms.get_transform(image).apply_image(image)
        return {
            "image": torch.as_tensor(input.astype("float32").transpose(2, 0, 1)).to(self.device),
            "height": height,
            "width": width
        }
    

    def forward(self, input):
        """
        The `forward` function takes an input, passes it through a model without gradient computation, and
        returns the outputs.
        
        :param input: The `input` parameter in the `forward` method is typically the input data that will be
        passed through the neural network model for inference or training. This input data could be a single
        data point, a batch of data points, or any other form of input that the model is designed to process
        :return: the outputs generated by passing the input through the model.
        """
        
        with torch.no_grad():
            outputs = self.model(input)
        return outputs
    
    def postprocess(self):
        pass

    def predict(self, image):
        """
        The `predict` function preprocesses an image and then passes it through a neural network for forward
        propagation to make a prediction.
        
        :param image: The `image` parameter is the input image that will be passed to the `predict` method
        for making predictions
        :return: The `predict` method is returning the output of the `forward` method applied to the
        preprocessed input image.
        """
        input = self.preprocess(image)
        return self.forward(input)


if __name__ == "__main__":
    import argparse
    import glob
    import cv2
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, help="Path to the weights file")
    parser.add_argument("--config-file", type=str, help="Path to the config file")
    parser.add_argument("--input-path", type=str, help="Path to the input folder")
    args = parser.parse_args()
    model = Detectron2Model(weights_path=args.weights, config_file=args.config_file)
    model.warmup()
    
    images = glob.glob(args.input_path + "/*") # TODO change this to the correct extension

    for image_path in images:
        image = cv2.imread(image_path, -1)
        outputs = model.predict(image)
        print(outputs)
        break
