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
[base working directory]\bboxes\n01440764\
[base working directory]\bboxes\n01443537\
[base working directory]\bboxes\n01484850\
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

## Step 3: Extracting and Reorganize the Validation Data
First, you will need to extract the contents of the file you downloaded that is named "ILSVRC2012_img_val.tar" into a sub directory of the base working directory called "ILSVRC2012_img_val".
Once that has been extracted into the subdirectory, you will find 50,000 validation images in the following directory:
```
[base working directory]\ILSVRC2012_img_val\
```
Rather than having all of these files in this one directory, it would be nice if they were moved into directories corresponding to their classes like the training data.

Once you these images are in place, you will want to execute the script named "reorg_validation_data.py" in the imagenet_prep_scripts folder from this repository, specifying the base working directory:
```
python reorg_validation_data.py \
  --base_working_dir=[base working directory]  
```
Note that this assumes you are executing the script with a working directory of the imagenet_prep_scripts folder from this repository.  Within that folder is a file named "imagenet_2012_validation_synset_labels.txt" that is needed by the script.  If you execute the script with a different working directory or have the "imagenet_2012_validation_synset_labels.txt" file in some other directory, you can specify it's location by passing it to the script in the `--validation_synset_labels_file` parameter.

## Step 4: Extract the Training Data
...

## Step 5: Build TFRecord files
...
