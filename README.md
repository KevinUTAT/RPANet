# RPANet
A PyTorch Yolo model for tracking Remotely Piloted Aircraft(RPA) \
The YOLOv5 implimentation is from Ultralytics LLC, and you can find the original model here: \
https://github.com/ultralytics/yolov5 \
\
The oldder model uses Yolov3 implimentation by Chris Fotache \
His original model can be find here: \
https://github.com/cfotache/pytorch_custom_yolo_training \
\
A more streamlined process is added to process training data coming in as \
videos and small modifications are made to the model to fit it for Drone tracking. \
\
To track drones in realtime, the Simple Online Realtime Tracker by Alex Bewley is used: \
https://github.com/abewley/sort


## Change log
### 2020-08-09:
Add a globle dictionary to store all tracking information. \
Add object motion tracking ability. \
The tracking ID is now displayed with the BBox. \
Speed of object (pixel speed) is now avaliable in the Drone class. \
Imporved active leaarning logic to reduce redundant images output.
### 2020-08-01:
Trying to incroperating **active learning**. \
The inference module will now (optional) save images that have detected target with low confidence. \
User should go label those images and add them into training dataset. \
This should improve the model with only small amout of labeling needed.
```
python detcet.py --scource screen --active 0.3
```
The above example will save images with detection confidence between 0.1 and 0.3.
### 2020-07-27:
Re-work the screen capture method. Using python-mss instead of PIL. \
This is able to increas inference speed to 15 to 20 fps (RTX2070 Max-Q), \
and reduce latency to around 0.1s 
### 2020-07-26:
Porting the latest v2.0 version of Ultralytics YOLOv5 in replace of v3. \
The weight file size is reduced to under 100MB, and seen small inference speed improvment.\
So far seems like the inference robustness also improved as scenes with small obstacles are beening recognized (wire fence) \
Tried first with YOLOv4 but is much more difficult to port the model \
compare to v5 as Ultralytics resued many code from v3 to implement v5
### 2020-07-10:
Including screen cap input. \
Turns out, its very dificult to get my HikVision cameras sets streaming API to work. So instead, a temperary solution is to have the streaming webview open and screen cap to our model. 

### 2020-07-07: 
New model implimentation from Ultralytics with much improved prefromence. 
### 2020-07-05: 
Upload current model to github
## Know issues
- BBox.py works quite poorly with 1920x1080 images, using makesense.ai instead 
## To do
- Bench marking performence between different model (S vs M vs L vs X)
- Find more applications?
  
## Files
**train.py**: Training the modle. Options can be set through arguments \
**detect.py**: Run tracking on vedios. Source can be set trough arguments \
**framer.py**: Get images from a video with a interval \
**rename.py**: rename images to unique name \
**train_val_spliter.py**: Dividing datas by set up *train.txt* and *val.txt* \
**look_at_my_screen.py**: A small toy to test screen cap performence

## Inference
### From a camera connected to the computer:
```
python detect.py --source 0
```
Replace 0 with the camera number if the camera is not the defult webcam. 
### From an IP camera:
If your IP camera require a difficult API to access its stream, you would be better to try using creen cap. \
If you can get it's stream in *http* or *rtsp* then try the following:
```
python detect.py --source rtsp://<username>:<password>@<IP address of device>:<RTSP port>/Streaming/channels/<channelnumber><stream number>
```
### From your screen:
If none of the above stream sources are possiable, we also support screen cap. \
If you can get your stream to play on your screen
```
python detcet.py --scource screen
```
This will start screen cap as shown below: 
```
1/1: screen... Plase select regin:
X:                                     1151Y:                                     1640
```
 Left click top left and then bottom right to select the screen regin you would like to capture.
