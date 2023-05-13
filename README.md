# YOLOv8 inference using Rust

This is a web interface to [YOLOv8 object detection neural network](https://ultralytics.com/yolov8)
implemented on [Rust](https://www.rust-lang.org/).

This is a source code for a ["How to create YOLOv8-based object detection web service using Python, Julia, Node.js, JavaScript, Go and Rust"](https://dev.to/andreygermanov/how-to-create-yolov8-based-object-detection-web-service-using-python-julia-nodejs-javascript-go-and-rust-4o8e) tutorial.

## Install

* Clone this repository: `git clone git@github.com:AndreyGermanov/yolov8_onnx_rust.git`

Ensure that the ONNX runtime installed on your operating system, because the library that integrated to the 
Rust package may not work correctly. To install it, you can download the archive for your operating system 
from [here](https://github.com/microsoft/onnxruntime/releases), extract and copy contents of "lib" subfolder
to the system libraries path of your operating system.

## Run

Execute:

```
cargo run
```

It will start a webserver on http://localhost:8080. Use any web browser to open the web interface.

Using the interface you can upload the image to the object detector and see bounding boxes of all objects detected on it.