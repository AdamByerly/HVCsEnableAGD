# Homogeneous Vector Capsules Enable Adaptive Gradient Descent in Convolutional Neural Networks

This repository contains the code used for the experiments detailed in a paper currently submitted to IEEE Transactions on Neural Networks and Learning Systems.  The paper is available pre-published at arXiv: https://arxiv.org/...

## Required Libraries
-   TensorFlow (see  [http://www.tensorflow.org](http://www.tensorflow.org/) 
-   NumPy (see  [http://www.numpy.org/](http://www.numpy.org/))
-   OpenCV if using the `--log_annotated_images` parameter (see below and see [http://opencv.org/](http://opencv.org/))
-   At least one GPU

## Reproducing the Results

We have provided 7 files, all named named with the prefix "train_" that correspond to the 7 experiments discussed in the paper:

 - train_simple.py
 - train_simple_hvc.py
 - train_inception_v3_baseline_1.py
 - train_inception_v3_baseline_2.py
 - train_inception_v3_baseline_w_Adam.py
 - train_inception_v3_hvc_defaults.py
 - train_inception_v3_hvc_decaying

They all take the same parameters, which I hope makes it easier for you to use them:
```
--data_dir
--black_list_file
--gpus
--batch_size
--start_epoch
--end_epoch
--run_name
--log_dir
--weights_file
--validate_all
--validate_nbl
--profile_compute_time_every_n_steps
--log_annotated_images
--save_summary_info_every_n_steps
--image_size
```
Further, all but the first 4 can safely be ignored when initiating an experiment.  When resuming a preemptively halted experiment, you will also need to address `--start_epoch`, `--run_name`, and `--weights_file`.
See below for more information on each parameter.

## Parameters
```
--data_dir
``` 
 **Required**.
 Use this to specify the location of the processed ImageNet data files.  This should be a directory that contains 1024 training and 128 validation files, respectively.  The files will match the patterns `train-????-of-1024` and `validation-?????-of-00128`, respectively.  See "Data Preparation" below to learn how to create these files.
```
--black_list_file
``` 
 **Required** (sort of, and only for the Inception v3 experiments).
 Use this to specify the location of the ImageNet blacklist file.  See "Data Preparation" below to learn more about this file.  Technically, you don't have to specify this file, and the code will still work, but when validating the Inception v3 experiments, two validations are run.  One for all images in the validation set and a second for those that haven't been deemed as blacklisted.  If you don't provide the location of this file, it will perform two identical validations, neither of which will exclude any images in the validation set.  The simple monolithic CNN experiments do not perform a separate validation pass on the non-blacklisted subset of the validation data, so it is ignored for those experiments.  See also `--validate_all` and `--validate_nbl`
```
--gpus
```
**Required** (sort of).  Default Value: 2
If you have exactly two GPUs that you want to use, then this parameter can be omitted.  Note that at least one GPU is required.
```
--batch_size
```
**Required** (sort of).  Default Values: 96 (for Inception v3 experiments); 128 (for simple monolithic CNN experiments)
The batch size you can use is limited by the memory available on your GPUs.  If you have 2 GeForce GTX 1080 Tis, which have 11GB of RAM each--like I did--, then you can omit this parameter.  If you have a different number of GPUs and/or a different amount of RAM on them, you will need to supply an appropriate value for this parameter.  In my experiments, I discovered that for the simple monolithic CNN, a batch size of ~5.82 per GB of GPU RAM per GPU is the threshold for the images and model to fit in memory.  For the Inception v3 experiments, a batch size of ~4.36 per GB of GPU RAM per GPU is the threshold for the images and model to fit in memory.  Note that the batch size must be evenly divisible by the number of GPUs being used.
```
--start_epoch
```
Optional.  Default Value: 1
You would only want to override this value if you were resuming training a model that was stopped for some reason.  In that case, you would set this value to the number of the next epoch you want to train relative to the epoch after which the weights you are starting with were saved. (see `--weights_file` below)
```
--end_epoch
```
Optional.  Default Values: 175 (for Inception v3 experiments); 350 (for simple monolithic CNN experiments)
Model training will continue until this many epochs have run, or something else stops it preemptively.
```
--run_name
```
Optional.  Default Value: an amalgamation of the digits taken from the current date and time.
For example:  20181115153357.  This run was created on November 15, 2018 at 3:33:57 PM local time.  When resuming a previously halted experiment, you will want to provide the run name that was used for that experiment in this parameter.
```
--log_dir
```
Optional.  Default Value: "logs"
The default value is a relative path which will be a subdirectory to the working directory from which you run the command.
```
--weights_file
```
Optional.  Default Value: None.
In the event that you want to restart a previously interrupted training session, you'll need to provide the last saved weights as a starting point.  Note that when you provide a weights file, you'll also want to set the --start_epoch parameter to the next epoch following the epoch for which the weights were saved in the specified file.
```
--validate_all
```
Optional.  Default Value: True.
Set to False to skip doing a validation pass after each epoch on the whole ImageNet validation set.  You will want either this, `--validate_nbl`, or both to be True.  This value is ignored for the simple monolithic CNN experiments. 
```
--validate_nbl
```
Optional.  Default Value: True.
Set to False to skip doing a validation pass after each epoch on the the non-blacklisted subset of the ImageNet validation set.  You will want either this, `--validate_all`, or both to be True.  This value is ignored for the simple monolithic CNN experiments. 
```
--profile_compute_time_every_n_steps
```
Optional.  Default Value: None
If you are using different GPUs or a different number of GPUs or are otherwise curious, you can set this flag to some positive integer.  100 is my suggestion.  This will cause tensorflow to profile the compute time used by the devices (CPUs and GPUs) on your machine each time this number of training steps has passed.  You can then look in TensorBoard, under the Graphs tab, and see a list of Session runs to choose from (one for each time the compute time was profiled).  You can then examine how compute time is being used during training for each of your devices.
```
--log_annotated_images
```
Optional.  Default Value: False.
When True, a random subset of images that will be used for training are logged as summary images to be viewed in TensorBoard.  This can be used to give you insight into the preprocessing that is done on the images before they are trained on.  See `--save_summary_info_every_n_steps`.
```
--save_summary_info_every_n_steps
```
Optional.  Default Value: None
This parameter works in conjunction with the `--log_annotated_images` as well as with any histogram values you may want to add if you are diving deep into these networks.  Set to a positive integer that represents how often you want to save the annotated images or histogram values.  Note that saving these values both takes disk space and slows down the training a little, so you would likely only want to use this exploratorily.  See [https://www.tensorflow.org/guide/tensorboard_histograms](https://www.tensorflow.org/guide/tensorboard_histograms) for more on adding histogram value saving to a network.
```
--image_size
```
Optional.  Default Value: 299 (for Inception v3 experiments); 224 (for simple monolithic CNN experiments).  This value should not be overridden unless you are changing the network design and doing your own experiments.

## Example command
```
python train_simple_hvc.py \
  --data_dir=C:\Users\adam\Downloads\ILSVRC2017_CLS-LOC\\Data\\CLS-LOC\\processed \
  --gpus=2 \
  --batch_size=128
```
Note: Results, including:
- a summary of progress at each epoch in a CSV
- TensorBoard related data and weights

will be saved in the `--log_dir` directory.

To view progress and the graph structure in TensorBoard, run the following command from the same working directory from which you launched the training script:
```
tensorboard --logdir ./logs
```
Once TensorBoard is running, direct your web browser to [http://localhost:6006](http://localhost:6006/).

See more about TensorBoard here:  [https://www.tensorflow.org/guide/summaries_and_tensorboard](https://www.tensorflow.org/guide/summaries_and_tensorboard)

# Data Preparation

...still working on this...

# A note on the excessive code duplication in this repository
As a software engineer, it pains me to see the same code only slightly altered copied and pasted into multiple places.  However, as a researcher, I want to spend more time researching and less time refactoring and testing code that will not be used "in production", so to speak.  You will see very similar code in this repository duplicated 7 times in the files that have names beginning with "train_".  As well as very similar code in 4 files beginning with "model_" and in two files named "output.py".
I have rationalized to myself that leaving this code duplication as it is will allow those who want to compare the different experiments or models to easily pull up the two files in their favorite diffing tool and be able to see exactly what is the same and what is different across any of the 7 experiments or any of the 4 models.

&nbsp;
&nbsp;

Maintained by Adam Byerly (abyerly@fsmail.bradley.edu)