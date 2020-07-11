# RPANet
A PyTorch Yolo model for tracking Remotely Piloted Aircraft(RPA) \
The YOLOv3 implimentation is from Ultralytics LLC, and you can find the original model here: \
https://github.com/roboflow-ai/yolov3 \
\
The old model uses Yolov3 implimentation by Chris Fotache \
His original model can be find here: \
https://github.com/cfotache/pytorch_custom_yolo_training \
\
A more streamlined process is added to process training data coming in as \
videos and small modifications are made to the model to fit it to Drone tracking.

## Change log
### 2020-07-10:
Including screen cap input. \
Turns out, its very dificult to get my HikVision cameras sets streaming API to work. So instead, a temperary solution is to hav the streaming webview open and screen cap to our model. \
```
python detcet.py --scource screen
```
This will start screen cap. Left click top left and then bottom right to select the screen regin
### 2020-07-07: 
New model implimentation from Ultralytics with much improved prefromence. 
### 2020-07-05: 
Upload current model to github
## Know issues
- BBox works quite poorly with 1920x1080 images, using makesense.ai instead /
## To do
- Intergration IP Camera stream in to the model
- YOLOv4??
## Files
**train_apex.py**: Training the modle. Options can be set through arguments \
**detect.py**: Run tracking on vedios. Source can be set trough arguments \
**framer.py**: Get images from a video with a interval \
**rename.py**: rename images to unique name \
**train_val_spliter.py**: Dividing datas by set up *train.txt* and *val.txt*
## Download
I have trina the model on 429 images. \
The pre-trained weight can be download here: \
https://drive.google.com/file/d/1HYVO936agc7nIoXqWBOVmsR46_IfebCU/view?usp=sharing \
This weight should be named *best.pt* and placed in folder *weights*

