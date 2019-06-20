# Data Preparation

## Step 1: Download Everything

First, if you do not have an image-net account, go to: http://www.image-net.org/signup.  After you have an account, you will need to login.  Go to: http://www.image-net.org/login

Once logged in you will be able to download the training and validation images and bounding boxes:
 1. Go to: http://www.image-net.org/download
 2. Select "Download Original Images (for non-commercial research/educational use only)" and click "Next"
 3. Find "Download links to ILSVRC2012 image data" on the page and click the link for that.
 4. Download: Training images (Task 1 & 2). 138GB.
     MD5: 1d675b47d978889d74fa0da5fadfb00e
 5. Download: Validation images (all tasks). 6.3GB.
     MD5: 29b22e2961454d5413ddabcf34fc5622
 6. Download: Training bounding box annotations (Task 1 & 2 only). 20MB.
     MD5: 9271167e2176350e65cfe4e546f14b17

Move all 3 files into a base working directory of your choice.  All of the steps below will be relative to this base working directory.  And when examples show `[base working directory]`, replace that with the directory you chose.
	
## Step 2: Create Bounding Boxes CSV
First, you will need to extract the contents of the file you downloaded that is named "ILSVRC2012_bbox_train_v2.tar.gz" into a sub directory of the base working directory called "bboxes".
Once that has been extracted into the subdirectory, you will find 1,000 subdirectories within that subdirectory (one for each class).  Within each one of those directories will be multiple .xml files.
Your directory structure should look like this:
```
[base working directory]\bboxes\n01440764
[base working directory]\bboxes\n01443537
[base working directory]\bboxes\n01484850
.
.
.
```
Once you these .xml files are in place, you will want to execute the script named "gen_train_bboxes.py" in the imagenet_prep_scripts folder from this repository, specifying the base working directory:
```
python gen_train_bboxes.py \
  --base_working_dir=[base working directory]
```
This process will create a file named "bboxes.csv" in the base working directory.  This process can take quite some time, depending on your hardware.

## Step 3: Reorganize the Validation Data
...

## Step 4: Build TFRecord files
...
