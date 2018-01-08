# StackGAN-v2
tensorflow version
I can't find any existed code implemented in tensorflow, so I try to implement one.
Download COCO data set and save the image name, caption and (width, height) by numpy. Then save the image and the caption into 
tfrecords file. 
Then run 
`python text_to_image.py --train` 
to train.
