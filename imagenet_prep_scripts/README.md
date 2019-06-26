# Data Preparation
The following instructions involve a .tar.gz file and many .tar files.  On Linux/Unix-based systems, you will have the ``tar`` command available to you to help with this.  On Windows-based systems, support for these file types is not built-in, so you will need an extraction tool.  I suggest 7-Zip (see: [https://www.7-zip.org/](https://www.7-zip.org/)).

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

Download all 3 files into a base working directory of your choice.  All of the steps below will be relative to this base working directory.  And when examples show `[base working directory]`, replace that with the directory you chose.
	
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
Once these .xml files are in place, you will want to execute the script named "gen_train_bboxes.py" in the imagenet_prep_scripts folder from this repository, specifying the base working directory:
```
python gen_train_bboxes.py --base_working_dir=[base working directory]
```
This process will create a file named "bboxes.csv" in the base working directory.  This process can take quite some time, depending on your hardware.

## Step 3: Extracting and Reorganize the Validation Data
First, you will need to extract the contents of the file you downloaded that is named "ILSVRC2012_img_val.tar" into a sub directory of the base working directory called "ILSVRC2012_img_val".
Once that has been extracted into the subdirectory, you will find 50,000 validation images in the following directory:
```
[base working directory]\ILSVRC2012_img_val\
```
Rather than having all of these files in this one directory, it would be nice if they were moved into directories corresponding to their classes like the training data.  So, once these images are in place, you will want to execute the script named "reorg_validation_data.py" in the imagenet_prep_scripts folder from this repository, specifying the base working directory:
```
python reorg_validation_data.py --base_working_dir=[base working directory]  
```
Note that this assumes you are executing the script with a working directory of the imagenet_prep_scripts folder from this repository.  Within that folder is a file named "imagenet_2012_validation_synset_labels.txt" that is needed by the script.  If you execute the script with a different working directory or have the "imagenet_2012_validation_synset_labels.txt" file in some other directory, you can specify it's location by passing it to the script in the `--validation_synset_labels_file` parameter.

Because all it needs to do is move 50,000 images into directories based on their classes, the script is relatively quick.  After executing your directory structure for the validation images should look like this:
```
[base working directory]\ILSVRC2012_img_val\n01440764\
[base working directory]\ILSVRC2012_img_val\n01443537\
[base working directory]\ILSVRC2012_img_val\n01484850\
.
.
.
```

## Step 4: Extract the Training Data
First, you will need to extract the contents of the file you downloaded that is named "ILSVRC2012_img_train.tar" into a sub directory of the base working directory called "ILSVRC2012_img_train".
Once that has been extracted into the subdirectory, you will find 1,000 additional tar files (one for each class) in the following directory:
```
[base working directory]\ILSVRC2012_img_train\
```
This process can take quite some time, depending on your hardware.

Now, each of those need to be extracted into separate directories named the same as each of the tar files (minue the .tar extension).  After doing so, your directory structure for the training images should look like this:
```
[base working directory]\ILSVRC2012_img_train\n01440764\
[base working directory]\ILSVRC2012_img_train\n01443537\
[base working directory]\ILSVRC2012_img_train\n01484850\
.
.
.
```
This process can take quite some time, depending on your hardware.

## Step 5: Build TFRecord files
The goal of this step is to convert the training and evaluation images into  
a sharded data set consisting of 1024 and 128 TFRecord files, respectively.  It is these files that are used during training (and validation) and not the actual/original image files.
In order to do this, you will want to execute the script named "build_imagenet_data.py" in the imagenet_prep_scripts folder from this repository, specifying the base working directory
```
python build_imagenet_data.py --base_working_dir=[base working directory]  
```
Note that this assumes you are executing the script with a working directory of the imagenet_prep_scripts folder from this repository. 
- Within that folder is a file named "imagenet_lsvrc_2015_synsets.txt" that is needed by the script.  If you execute the script with a different working directory or have the "imagenet_lsvrc_2015_synsets.txt" file in some other directory, you can specify its location by passing it to the script in the `--labels_file` parameter.
- Within that folder is a file named "imagenet_metadata.txt" that is needed by the script.  If you execute the script with a different working directory or have the "imagenet_metadata.txt" file in some other directory, you can specify its location by passing it to the script in the `--imagenet_metadata_file` parameter.

Also note that by default there are 8 threads designated to do this work.  You can override this with the `--num_threads` parameter.

This script may take several hours or more, depending on your hardware.

After this has finished, you will use the following directory for the `--data_dir` parameter used by each of the training scripts in the root folder of this repository.
```
[base working directory]\processed\
```

After it is finished, if you are running low on disk space, only the TFRecord files in the "processed" directory are needed, so you can consider deleting the bounding box files, the .tar files and images that were used to create the TFRecord files.
