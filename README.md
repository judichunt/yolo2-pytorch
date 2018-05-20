# YOLOv2 for real-time video Object Detection
This is modified from the [PyTorch](https://github.com/pytorch/pytorch)
implementation of YOLOv2(by Long Chen https://github.com/longcw/yolo2-pytorch).    
For the video process, I used [OpenCV](https://github.com/opencv/opencv/tree/3.4.1) and [FFmpeg](https://www.ffmpeg.org/) for real-time and frame-based transformation.         
Real-time Object Detection works on 720P videos of different types.

The model is mainly based on [darkflow](https://github.com/thtrieu/darkflow)
and [darknet](https://github.com/pjreddie/darknet).

Used a Cython extension for postprocessing and 
`multiprocessing.Pool` for image preprocessing.
Testing an image in VOC2007 costs about 13~20ms.

For details about YOLO and YOLOv2 please refer to their [project page](https://pjreddie.com/darknet/yolo/) 
and the [paper](https://arxiv.org/abs/1612.08242):
*YOLO9000: Better, Faster, Stronger by Joseph Redmon and Ali Farhadi*.
## Requirements
* python 3.6
* Anaconda3
* pytorch 0.3.0+
* [gcc](https://anaconda.org/anaconda/gxx_linux-64)
* cuda 8.0+
    


## Installation and apply Object Detection on Videos
1. Clone this repository
    ```bash
    git clone git@github.com:longcw/yolo2-pytorch.git
    ```

2. Build the reorg layer ([`tf.extract_image_patches`](https://www.tensorflow.org/api_docs/python/tf/extract_image_patches))
    ```bash
    cd yolo2-pytorch
    ./make.sh
    ```
3. Install opencv
    ```bash
    conda install -c conda-forge opencv 
    ```

4. Download the trained model [yolo-voc.weights.h5](https://drive.google.com/open?id=0B4pXCfnYmG1WUUdtRHNnLWdaMEU)       
   Set the model path `trained_model `, and `input_dir` `filename` of video file(.avi, .mpeg, ...) in `realtime_OD.py`

5. Run `python realtime_OD.py`. The real-time Object Dectection video will be played as running the model.                   

   And the output video will be saved as the same type as your input video.

## Demo video
![image](https://github.com/judichunt/yolo2-pytorch-realtime-video/blob/master/demo_gif/example.gif)
![image](https://github.com/judichunt/yolo2-pytorch-realtime-video/blob/master/demo_gif/Obj_HomeOffice.gif)
![image](https://github.com/judichunt/yolo2-pytorch-realtime-video/blob/master/demo_gif/Obj_HomeOffice1.gif)
![image](https://github.com/judichunt/yolo2-pytorch-realtime-video/blob/master/demo_gif/Obj_TaipeiStreet.gif)
![image](https://github.com/judichunt/yolo2-pytorch-realtime-video/blob/master/demo_gif/Obj_TaipeiStreet1.gif)

## Training YOLOv2, Evaluation, Training on your own data
Follow the instructions in            
[YOLOv2 in PyTorch](https://github.com/longcw/yolo2-pytorch).
