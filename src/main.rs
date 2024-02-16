use std::{sync::Arc, path::Path, vec};
use image::{GenericImageView, imageops::FilterType};
use ndarray::{Array, IxDyn, s, Axis};
use ort::{Environment,SessionBuilder,Value};
use rocket::{response::content,fs::TempFile,form::Form};
#[macro_use] extern crate rocket;

// Main function that defines
// a web service endpoints a starts
// the web service
#[rocket::main]
async fn main() {
    rocket::build()
        .mount("/", routes![index])
        .mount("/detect", routes![detect])
        .launch().await.unwrap();
}

// Site main page handler function.
// Returns Content of index.html file
#[get("/")]
fn index() -> content::RawHtml<String> {
    content::RawHtml(std::fs::read_to_string("index.html").unwrap())
}

// Handler of /detect POST endpoint
// Receives uploaded file with a name "image_file", passes it
// through YOLOv8 object detection network and returns and array
// of bounding boxes.
// Returns a JSON array of objects bounding boxes in format [(x1,y1,x2,y2,object_type,probability),..]
#[post("/", data = "<file>")]
fn detect(file: Form<TempFile<'_>>) -> String {
    let buf = std::fs::read(file.path().unwrap_or(Path::new(""))).unwrap_or(vec![]);
    let boxes = detect_objects_on_image(buf);
    return serde_json::to_string(&boxes).unwrap_or_default()
}

// Function receives an image,
// passes it through YOLOv8 neural network
// and returns an array of detected objects
// and their bounding boxes
// Returns Array of bounding boxes in format [(x1,y1,x2,y2,object_type,probability),..]
fn detect_objects_on_image(buf: Vec<u8>) -> Vec<(f32,f32,f32,f32,&'static str,f32)> {
    let (input,img_width,img_height) = prepare_input(buf);
    let output = run_model(input);
    return process_output(output, img_width, img_height);
}

// Function used to convert input image to tensor,
// required as an input to YOLOv8 object detection
// network.
// Returns the input tensor, original image width and height
fn prepare_input(buf: Vec<u8>) -> (Array<f32,IxDyn>, u32, u32) {
    let img = image::load_from_memory(&buf).unwrap();
    let (img_width, img_height) = (img.width(), img.height());
    let img = img.resize_exact(640, 640, FilterType::CatmullRom);
    let mut input = Array::zeros((1, 3, 640, 640)).into_dyn();
    for pixel in img.pixels() {
        let x = pixel.0 as usize;
        let y = pixel.1 as usize;
        let [r,g,b,_] = pixel.2.0;
        input[[0, 0, y, x]] = (r as f32) / 255.0;
        input[[0, 1, y, x]] = (g as f32) / 255.0;
        input[[0, 2, y, x]] = (b as f32) / 255.0;
    };
    return (input, img_width, img_height);
}

// Function used to pass provided input tensor to
// YOLOv8 neural network and return result
// Returns raw output of YOLOv8 network
fn run_model(input:Array<f32,IxDyn>) -> Array<f32,IxDyn> {
    let env = Arc::new(Environment::builder().with_name("YOLOv8").build().unwrap());
    let model = SessionBuilder::new(&env).unwrap().with_model_from_file("yolov8m.onnx").unwrap();
    let input_as_values = &input.as_standard_layout();
    let model_inputs = vec![Value::from_array(model.allocator(), input_as_values).unwrap()];
    let outputs = model.run(model_inputs).unwrap();
    let output = outputs.get(0).unwrap().try_extract::<f32>().unwrap().view().t().into_owned();
    return output;
}

// Function used to convert RAW output from YOLOv8 to an array
// of detected objects. Each object contain the bounding box of
// this object, the type of object and the probability
// Returns array of detected objects in a format [(x1,y1,x2,y2,object_type,probability),..]
fn process_output(output:Array<f32,IxDyn>,img_width: u32, img_height: u32) -> Vec<(f32,f32,f32,f32,&'static str, f32)> {
    let mut boxes = Vec::new();
    let output = output.slice(s![..,..,0]);
    for row in output.axis_iter(Axis(0)) {
        let row:Vec<_> = row.iter().map(|x| *x).collect();
        let (class_id, prob) = row.iter().skip(4).enumerate()
            .map(|(index,value)| (index,*value))
            .reduce(|accum, row| if row.1>accum.1 { row } else {accum}).unwrap();
        if prob < 0.5 {
            continue
        }
        let label = YOLO_CLASSES[class_id];
        let xc = row[0]/640.0*(img_width as f32);
        let yc = row[1]/640.0*(img_height as f32);
        let w = row[2]/640.0*(img_width as f32);
        let h = row[3]/640.0*(img_height as f32);
        let x1 = xc - w/2.0;
        let x2 = xc + w/2.0;
        let y1 = yc - h/2.0;
        let y2 = yc + h/2.0;
        boxes.push((x1,y1,x2,y2,label,prob));
    }

    boxes.sort_by(|box1,box2| box2.5.total_cmp(&box1.5));
    let mut result = Vec::new();
    while boxes.len()>0 {
        result.push(boxes[0]);
        boxes = boxes.iter().filter(|box1| iou(&boxes[0],box1) < 0.7).map(|x| *x).collect()
    }
    return result;
}

// Function calculates "Intersection-over-union" coefficient for specified two boxes
// https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/.
// Returns Intersection over union ratio as a float number
fn iou(box1: &(f32, f32, f32, f32, &'static str, f32), box2: &(f32, f32, f32, f32, &'static str, f32)) -> f32 {
    return intersection(box1, box2) / union(box1, box2);
}

// Function calculates union area of two boxes
// Returns Area of the boxes union as a float number
fn union(box1: &(f32, f32, f32, f32, &'static str, f32), box2: &(f32, f32, f32, f32, &'static str, f32)) -> f32 {
    let (box1_x1,box1_y1,box1_x2,box1_y2,_,_) = *box1;
    let (box2_x1,box2_y1,box2_x2,box2_y2,_,_) = *box2;
    let box1_area = (box1_x2-box1_x1)*(box1_y2-box1_y1);
    let box2_area = (box2_x2-box2_x1)*(box2_y2-box2_y1);
    return box1_area + box2_area - intersection(box1, box2);
}

// Function calculates intersection area of two boxes
// Returns Area of intersection of the boxes as a float number
fn intersection(box1: &(f32, f32, f32, f32, &'static str, f32), box2: &(f32, f32, f32, f32, &'static str, f32)) -> f32 {
    let (box1_x1,box1_y1,box1_x2,box1_y2,_,_) = *box1;
    let (box2_x1,box2_y1,box2_x2,box2_y2,_,_) = *box2;
    let x1 = box1_x1.max(box2_x1);
    let y1 = box1_y1.max(box2_y1);
    let x2 = box1_x2.min(box2_x2);
    let y2 = box1_y2.min(box2_y2);
    return (x2-x1)*(y2-y1);
}

// Array of YOLOv8 class labels
const YOLO_CLASSES:[&str;80] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
    "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
];
