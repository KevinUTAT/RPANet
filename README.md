# RPANet
A PyTorch Yolo model for tracking Remotely Piloted Aircraft(RPA) \
The model use Yolov3 implimentation by Chris Fotache \
His original model can be find here: \
https://github.com/cfotache/pytorch_custom_yolo_training \
A more streamlined process is added to process training data coming in as \
videos and small modifications are made to the model to fit it to Drone tracking.

## Change log
2020-07-05: Upload current model to github
## Know issues
- BBox works quite poorly with 1920x1080 images 
- The model only work with PNG images but changing to jpg are only minor changes
- Model performence don't seems to be the best, even it is CUDA enabled, a heavy amount of load is putted on the CPU
## files
train.py: Training the modle. Options can be set through arguments \
track.py: Run tracking on vedios. Have to set the video path in file as *videopath* \
framer.py: Get images from a video with a interval \
rename.py: rename images to unique name \
train_val_spliter: Dividing datas by set up *train.txt* and *val.txt*

