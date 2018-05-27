# Applied_Project
This repo is just for the neural network classification code for Applied Robotics Course.

This is a make it work once type of project (unfortunately). This repo is a mess; I just needed to backup so that nothing disastrous happens.

I installed the tf-openpose library. Inside ~/tf-openpose/src, I made these files.

- run_webcam_mod.py (this saves the coordinates)
- run_webcam_mod_predict.py (this makes predictions based on trained model)
These files are in the "backup_of..."

Inside ~/tf-openpose, I had made a different folder to put the pickled files, ~/tf-openpose/pickled. Here I wrote the code to train on the data.
The outer contents of this repo is the content of ~/tf-openpose/pickled.
- train.py trains the model and saves as a .pth file
