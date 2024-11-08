import os
import sys
import subprocess
import argparse


def convert_to_torchscript(**kwargs):
    subprocess.run(f"python {os.path.join(os.path.dirname(os.path.realpath(__file__)),'converter/detectron2_to_onnx.py')} --output /home/weights --format torchscript --export-method scripting", shell=True)

def convert_to_onnx(**kwargs):
    subprocess.run(f"python {os.path.join(os.path.dirname(os.path.realpath(__file__)),'converter/detectron2_to_onnx.py')}", shell=True)
    subprocess.run(f"python {os.path.join(os.path.dirname(os.path.realpath(__file__)),'converter/detectron2_onnx_trtonnx.py')} -b {kwargs.get('batch_size', 1)}", shell=True)

def convert_to_trt(**kwargs):
    command = f'trtexec --onnx={os.path.join(f"/home/weights/onnx", "exported.onnx")} --saveEngine={os.path.join(f"/home/weights/", "model.engine")} --useCudaGraph'
    if kwargs.get('fp16', False):
        command += ' --fp16'
        
    subprocess.run(command, shell=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert a model to ONNX and then to TensorRT.")
    parser.add_argument("-b","--batch-size", type=int, help="Batch size for the model", default=1)
    parser.add_argument("--fp16", '-fp16',action='store_true', help="Use fp16")
    parser.add_argument("--torchscript", '-ts',action='store_true', help="Convert to torchscript")
    parser.add_argument("--onnx", '-onnx',action='store_true', help="Convert to onnx")
    parser.add_argument("--trt", '-trt',action='store_true', help="Convert to TensorRT")
    args = parser.parse_args()
    if args.torchscript:
        convert_to_torchscript()
    if args.onnx:
        convert_to_onnx(batch_size=args.batch_size)
    if args.trt:
        convert_to_onnx(batch_size=args.batch_size)
        convert_to_trt(fp16=args.fp16)