# YOLOv8 inference using Rust

This is a web interface to [YOLOv8 neural network](https://ultralytics.com/yolov8)
implemented on [Rust](https://www.rust-lang.org/) using ONNX.

This is a source code for a ["How to create YOLOv8-based object detection web service using Python, Julia, Node.js, JavaScript, Go and Rust"](https://dev.to/andreygermanov/how-to-create-yolov8-based-object-detection-web-service-using-python-julia-nodejs-javascript-go-and-rust-4o8e) tutorial
extended by instance segmentation. This web application uses the YOLOv8 both for object detection and segmentation.

See demo here: 

The algorithm, that used to run YOLOv8 segmentation using ONNX described in [this article](https://dev.to/andreygermanov/how-to-implement-instance-segmentation-using-yolov8-neural-network-3if9).


## Install

* Clone this repository: `git clone git@github.com:AndreyGermanov/yolov8_onnx_rust_segmentation.git`

Ensure that the ONNX runtime installed on your operating system, because the library that integrated to the 
Rust package may not work correctly. To install it, you can download the archive for your operating system 
from [here](https://github.com/microsoft/onnxruntime/releases), extract and copy contents of "lib" subfolder
to the system libraries path of your operating system.

* Download the [yolov8m_seg.onnx](https://drive.google.com/file/d/1uG1nagxQoyvcHfUYXcNDXJCvglsz7rdT/view?usp=sharing) model file and put it to the root of downloaded repository. 
## Run

Execute:

```
cargo run
```

It will start a webserver on http://localhost:8080. Use any web browser to open the web interface.

Using the interface you can upload the image to the object detector and see bounding boxes and segmentation masks of 
all objects detected on it.