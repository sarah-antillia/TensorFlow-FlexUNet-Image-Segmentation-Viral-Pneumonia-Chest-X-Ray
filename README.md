<h2>TensorFlow-FlexUNet-Image-Segmentation-Viral-Pneumonia-Chest-X-Ray (2026/11/18)</h2>
<h3>Viral-Pneumonia-Chest-X-Ray: AI Generated Pseudo Masks Segmentation Challenge</h3>
Sarah T. Arai<br>
Software Laboratory antillia.com<br><br>
This is the first experiment of Image Segmentation for <b>Viral-Pneumonia-Chest-X-Ray </b>
 based on
our <a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model">TensorFlowFlexUNet Model</a>
 (<b>TensorFlow Flexible UNet Image Segmentation Model for Multiclass</b>) and a 512x512-pixel PNG
<a href="https://drive.google.com/file/d/1iXmJqBpHZTLhsTdqerQV8J4RBzdIYUJS/view?usp=sharing">
<b>Augmented-Viral-Pneumonia-Chest-X-Ray-ImageMask-Datasaet.zip</b></a>
with colorized masks (<a href="https://www.mit.edu/~amini/LICENSE.md">MIT</a>
) derived  by us from the Kaggle
<br><br>
<a href="https://www.kaggle.com/datasets/divyam6969/chest-xray-pneumonia-dataset">
<b>Chest XRAY - Pneumonia Dataset
</b>
</a> by Divyam6969.
<br><br>
<hr>
<b>Actual Image Segmentation for Viral-Pneumonia-Chest-X-Ray Images of 512x512 pixels</b><br>
As shown below, the inferred masks resemble the ground-truth masks. <br>
<br>
<b>class_color_map = {Lung: cyan, Pneumonia: dark_red}</b>
<br>
<br>
<table>
<tr>
<th width="320" height="auto">Input: image</th>
<th width="320" height="auto">Mask (ground_truth)</th>
<th width="320" height="auto">Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Viral-Pneumonia-Chest-X-Ray/mini_test/images/hflipped_person95_virus_177.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Viral-Pneumonia-Chest-X-Ray/mini_test/masks/hflipped_person95_virus_177.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Viral-Pneumonia-Chest-X-Ray/mini_test_output/hflipped_person95_virus_177.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Viral-Pneumonia-Chest-X-Ray/mini_test/images/hflipped_person1158_virus_1943.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Viral-Pneumonia-Chest-X-Ray/mini_test/masks/hflipped_person1158_virus_1943.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Viral-Pneumonia-Chest-X-Ray/mini_test_output/hflipped_person1158_virus_1943.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Viral-Pneumonia-Chest-X-Ray/mini_test/images/person379_virus_762.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Viral-Pneumonia-Chest-X-Ray/mini_test/masks/person379_virus_762.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Viral-Pneumonia-Chest-X-Ray/mini_test_output/person379_virus_762.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>
<h3>1. Dataset Citation</h3>
The dataset used here was derived from the following Kaggle:<br><br>
<a href="https://www.kaggle.com/datasets/divyam6969/chest-xray-pneumonia-dataset">
<b>Chest XRAY - Pneumonia Dataset
</b>
</a> by Divyam6969.
<br><br>
The following explanation (except) was taken from the website above. 
<br><br>
<b>About Dataset</b><br>
<b>Overview:</b><br>
The Pneumonia Chest X-ray Dataset is a collection of medical images designed for the development 
and evaluation of machine learning models in the detection and classification of
 pneumonia. The dataset is organized into two main folders: "train" and "test,"
 facilitating the training and testing phases of model development.
<br><br>
<b>Dataset Structure:</b><br>
<b>Train Folder:</b><br>
The "train" folder contains a comprehensive set of chest X-ray images that serve as the training 
data for machine learning models.<br>
Within the "train" folder, there are two distinct subfolders, each representing a different 
type of pneumonia:<br>
<ul>
<li>Bacterial Pneumonia:
 This subfolder contains chest X-ray images depicting cases of pneumonia 
caused by bacterial infections.
</li>
<li>Viral Pneumonia:
 Images in this subfolder showcase instances of pneumonia resulting from viral infections.
 </li>
</ul>
<!--
Fungal Pneumonia: The third subfolder includes chest X-ray images specifically related to pneumonia caused by fungal infections.
-->
<b>Test Folder:</b><br>
The "test" folder is dedicated to evaluating the performance of machine learning models on new, unseen data.
It includes chest X-ray images similar to those in the "train" folder but without any overlapping samples.
Key Features:

<br><br>
<b>Image Variety: </b><br>
The dataset comprises a diverse range of chest X-ray images, capturing various manifestations of pneumonia 
across different patients.
<br>
Two Pneumonia Types: The classification task involves distinguishing between bacterial and viral 
 pneumonia, enabling the development of models with specific diagnostic capabilities.
<br>
Real-world Relevance: The dataset reflects the complexities and challenges encountered in 
real-world medical imaging scenarios.
<br>
Balanced Distribution: Efforts have been made to maintain a balanced distribution of cases across 
the two pneumonia types, ensuring that the model is exposed to sufficient examples of 
each category during training.
<br><br>
<b>License</b><br>
<a href="https://www.mit.edu/~amini/LICENSE.md">MIT</a>
<br>
<br>
<h3>
2. Pneumonia-Chest-X-Ray ImageMask Dataset
</h3>
<h3>2.1 Download ImageMask Dataset</h3>
 If you would like to train this Viral-Pneumonia-Chest-X-Ray Segmentation model,
 please download the dataset from the Google Drive 
<a href="https://drive.google.com/file/d/1iXmJqBpHZTLhsTdqerQV8J4RBzdIYUJS/view?usp=sharing">
<b>Augmented-Viral-Pneumonia-Chest X-Ray-ImageMask-Dataset.zip</b>.
</a> 
Expand the downloaded and put it under the <b>./dataset</b> folder.
<br>
<pre>
./dataset
└─Viral-Pneumonia-Chest-X-Ray
    ├─test
    │   ├─images
    │   └─masks
    ├─train
    │   ├─images
    │   └─masks
    └─valid
        ├─images
        └─masks
</pre>
<br>
<b>Viral-Pneumonia-Chest-X-Ray Statistics</b><br>
<img src ="./projects/TensorFlowFlexUNet/Viral-Pneumonia-Chest-X-Ray/Viral-Pneumonia-Chest-X-Ray_Statistics.png" width="512" height="auto"><br>
<br>
As shown above, the number of images in the training and valid datasets is large enough to use for the
 training set of our segmentation model.
<br>
<h3>2.2 Derivation of ImageMask Dataset</h3>
The folder structure of the original train dataset is as follows, but it contains no annotation (mask) files.
<pre>
./Chest XRAY - Pneumonia Dataset
└─train
    ├─BACTERIAL  ( 2,530 files)
    │   ├─person1_bacteria_1.jpeg
...
    │   └─person1945_bacteria_4872.jpeg
    │
    ├─NORMAL     ( 1,341 files)
    │   ├─IM-0115-0001.jpeg
...	
    │   └─NORMAL2-IM-1423-0001.jpeg
    │	
    └─VIRAL      ( 1,345 files)
        ├─person80_virus_150.jpeg
...
        └─person1609_virus_2791.jpeg
</pre>
<b>Step 1</b><br>
We generated a 512x512-pixel cropped PNG master images from the JPEG files in 
<b>VIRAL</b> subfolder.
<br><br>
<b>Step 2</b><br>
We generated the first pseudo masks (Lung: cyan and Pneumonia: dark_red)
 corresponding to the master images
by applying a segmentation (inference) method of the pretrained model 
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Chest-X-Ray">
TensorFlow-FlexUNet-Image-Segmentation-Chest-X-Ray
</a> to the master images.
<br><br>
<b>Step 3</b><br>
We generated an augmented ImageMask dataset from all pairs of the first pseudo masks 
and the master images
by using an image flipping tool.
<br><br>
<b>Step 4</b><br>
We generated a new pretrained FlexUNet segmentation model by using the ImageMask 
dataset derived at Step 3.
<br><br>
<b>Step 5</b><br>
We generated the second pseudo masks corresponding to the master images by applying a segmentation method of the new 
pretrained FlexUNet model obtained at Step 4.
<br><br>
<b>Step 6</b><br>
We finally generated our own augmented ImageMask dataset from all pairs of the second pseudo masks and the master images
by using an image flipping tool.
<br><br>
<h3>2.3 Train Sample Images and Masks</h3>
<b>Train_sample_images</b><br>
<img src="./projects/TensorFlowFlexUNet/Viral-Pneumonia-Chest-X-Ray/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train_sample_masks</b><br>
<img src="./projects/TensorFlowFlexUNet/Viral-Pneumonia-Chest-X-Ray/asset/train_masks_sample.png" width="1024" height="auto">
<br>
<h3>
3. Train TensorFlowFlexUNet Model
</h3>
 We trained the Viral-Pneumonia-Chest-X-Ray TensorFlowFlexUNet model using the following
<a href="./projects/TensorFlowFlexUNet/Viral-Pneumonia-Chest-X-Ray/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorFlowFlexUNet/Viral-Pneumonia-Chest-X-Ray and run the following bat file.<br>
<pre>
>1.train.bat
</pre>
This runs the following command.<br>
<pre>
>python ../../../src/TensorFlowFlexUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>
<b>Model parameters</b><br>
Defined a small <b>base_filters=16 </b> and large <b>base_kernels=(11,11)</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet.py</a> 
and a large <b>num_layers=8</b> (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
; You may specify your own UNet class derived from our TensorFlowFlexModel
model         = "TensorFlowFlexUNet"
generator     =  False
image_width    = 512
image_height   = 512
image_channels = 3
num_classes    = 3
base_filters   = 16
base_kernels   = (11,11)
num_layers     = 8
dropout_rate   = 0.03
dilation       = (1,1)
</pre>
<b>Learning rate</b><br>
Defined a small learning rate.  
<pre>
[model]
learning_rate  = 0.00007
</pre>
<b>Loss and metrics functions</b><br>
Specified "categorical_focal_dice_loss" and <a href="./src/dice_coef_multiclass.py">"dice_coef_hybrid"</a>.<br>
<pre>
[model]
loss           = "categorical_focal_dice_loss"
metrics        = ["dice_coef_hybrid"]
hybrid_alpha   = 1.6
hybrid_beta    = 0.4
</pre>
<b>Dataset class</b><br>
Specified <a href="./src/ImageCategorizedMaskDataset.py">ImageCategorizedMaskDataset</a> class.<br>
<pre>
[dataset]
class_name    = "ImageCategorizedMaskDataset"
</pre>
<br>
<b>Learning rate reducer callback</b><br>
Enabled the learning_rate_reducer callback and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.4
reducer_patience   = 4
</pre>
<b>Early stopping callback</b><br>
Enabled early stopping callback with the patience parameter.
<pre>
[train]
patience      = 10
</pre>
<b>RGB Color map</b><br>
Specified RGB color map dict for Pneumonia-Chest-X-Ray 1+2 classes.<br>
<pre>
[mask]
mask_datatyoe    = "categorized"
mask_file_format = ".png"
;Viral-Pneumonia-Chest-X-Ray RGB color map dict for 1+2 classes.
; Background: black,  Lung:cyan,   Viral-Pneumonia:dark_red
rgb_map = {(0,0,0):0,(0,255,255):1, (210,20,20):2, }
</pre>
<b>Epoch change inference callback</b><br>
Enabled <a href="./src/EpochChangeInferencer.py">epoch_change_infer callback</a></b>.<br>
<pre>
[train]
epoch_change_infer       = True
epoch_change_infer_dir   =  "./epoch_change_infer"
num_infer_images         = 6
</pre>
By using this callback, on every epoch change, the inference procedure can be called
 for 6 images in the <b>mini_test</b> folder. This will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> 
<br> 
<!--
As shown below, early in the model training, the predicted masks from our FlexUNet segmentation model showed 
discouraging results.
 However, as training progressed through the epochs, the predictions gradually improved. 
 <br> 
<br>
-->
<b>Epoch_change_inference output at starting (epoch 1,2,3)</b><br>
<img src="./projects/TensorFlowFlexUNet/Viral-Pneumonia-Chest-X-Ray/asset/epoch_change_infer_at_start.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at middlepoint (epoch 13,14,15)</b><br>
<img src="./projects/TensorFlowFlexUNet/Viral-Pneumonia-Chest-X-Ray/asset/epoch_change_infer_at_middle.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at ending (epoch 27,28,29)</b><br>
<img src="./projects/TensorFlowFlexUNet/Viral-Pneumonia-Chest-X-Ray/asset/epoch_change_infer_at_end.png" width="1024" height="auto"><br>
<br>
In this experiment, the training process was stopped at epoch 29 by EarlyStoppingCallback.<br>
<img src="./projects/TensorFlowFlexUNet/Viral-Pneumonia-Chest-X-Ray/asset/train_console_output_at_epoch29.png" width="1024" height="auto"><br>
<br>
<a href="./projects/TensorFlowFlexUNet/Viral-Pneumonia-Chest-X-Ray/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/Viral-Pneumonia-Chest-X-Ray/eval/train_metrics.png" width="520" height="auto"><br>
<br>
<a href="./projects/TensorFlowFlexUNet/Viral-Pneumonia-Chest-X-Ray/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/Viral-Pneumonia-Chest-X-Ray/eval/train_losses.png" width="520" height="auto"><br>
<br>
<h3>
4. Evaluation
</h3>
Please move to <b>./projects/TensorFlowFlexUNet/Viral-Pneumonia-Chest-X-Ray</b> folder,
and run the following bat file to evaluate the TensorFlowUNet model for Pneumonia-Chest-X-Ray.<br>
<pre>
>./2.evaluate.bat
</pre>
This runs the following command.
<pre>
>python ../../../src/TensorFlowFlexUNetEvaluator.py ./train_eval_infer.config
</pre>
Evaluation console output:<br>
<img src="./projects/TensorFlowFlexUNet/Viral-Pneumonia-Chest-X-Ray/asset/evaluate_console_output_at_epoch29.png" width="1024" height="auto">
<br><br>Image-Segmentation-Pneumonia-Chest-X-Ray

<a href="./projects/TensorFlowFlexUNet/Viral-Pneumonia-Chest-X-Ray/evaluation.csv">evaluation.csv</a><br>
The loss (categorical_focal_dice_loss) on this Pneumonia-Chest-X-Ray/test was low, 
and dice_coef_hybrid was high,
 as shown below.
<br>
<pre>
categorical_focal_dice_loss,0.1188
dice_coef_hybrid,0.938
</pre>
<br>
<h3>
5. Inference
</h3>
Please move to <b>./projects/TensorFlowFlexUNet/Viral-Pneumonia-Chest-X-Ray</b> folder
and run the following bat file to infer segmentation regions for images using the trained TensorFlowUNet model for Pneumonia-Chest-X-Ray.<br>
<pre>
>./3.infer.bat
</pre>
This runs the following command.
<pre>
>python ../../../src/TensorFlowFlexUNetInferencer.py ./train_eval_infer.config
</pre>
<hr>
<b>mini_test_images</b><br>
<img src="./projects/TensorFlowFlexUNet/Viral-Pneumonia-Chest-X-Ray/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorFlowFlexUNet/Viral-Pneumonia-Chest-X-Ray/asset/mini_test_masks.png" width="1024" height="auto"><br>
<hr>
<b>Inferred test masks</b><br>
<img src="./projects/TensorFlowFlexUNet/Viral-Pneumonia-Chest-X-Ray/asset/mini_test_output.png" width="1024" height="auto"><br>
<br>
<hr>
<b>Enlarged images and masks for Viral-Pneumonia-Chest-X-Ray Images of 512x512 pixels</b><br>
As shown below, the inferred masks look similar to the ground truth masks except for the first and third cases.<br>
<br>
<b>class_color_map = {Lung: cyan, Pneumonia: dark_red}</b>
<br><br>
<table>
<tr>
<th width="320" height="auto">Image</th>
<th width="320" height="auto">Mask (ground_truth)</th>
<th width="320" height="auto">Inferred-mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Viral-Pneumonia-Chest-X-Ray/mini_test/images/hflipped_person124_virus_251.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Viral-Pneumonia-Chest-X-Ray/mini_test/masks/hflipped_person124_virus_251.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Viral-Pneumonia-Chest-X-Ray/mini_test_output/hflipped_person124_virus_251.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Viral-Pneumonia-Chest-X-Ray/mini_test/images/hflipped_person524_virus_1045.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Viral-Pneumonia-Chest-X-Ray/mini_test/masks/hflipped_person524_virus_1045.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Viral-Pneumonia-Chest-X-Ray/mini_test_output/hflipped_person524_virus_1045.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Viral-Pneumonia-Chest-X-Ray/mini_test/images/hflipped_person588_virus_1135.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Viral-Pneumonia-Chest-X-Ray/mini_test/masks/hflipped_person588_virus_1135.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Viral-Pneumonia-Chest-X-Ray/mini_test_output/hflipped_person588_virus_1135.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Viral-Pneumonia-Chest-X-Ray/mini_test/images/hflipped_person1381_virus_2375.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Viral-Pneumonia-Chest-X-Ray/mini_test/masks/hflipped_person1381_virus_2375.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Viral-Pneumonia-Chest-X-Ray/mini_test_output/hflipped_person1381_virus_2375.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Viral-Pneumonia-Chest-X-Ray/mini_test/images/person298_virus_617.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Viral-Pneumonia-Chest-X-Ray/mini_test/masks/person298_virus_617.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Viral-Pneumonia-Chest-X-Ray/mini_test_output/person298_virus_617.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Viral-Pneumonia-Chest-X-Ray/mini_test/images/person1080_virus_1791.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Viral-Pneumonia-Chest-X-Ray/mini_test/masks/person1080_virus_1791.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Viral-Pneumonia-Chest-X-Ray/mini_test_output/person1080_virus_1791.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>
<h3>
References
</h3>
<b>1. Efficient and Accurate Pneumonia Detection Using a Novel Multi-Scale Transformer Approach</b><br>
Alireza Saber, Pouria Parhami, Alimihammad Siahkarzadeh, Amirreza Fateh<br>
<a href ="https://arxiv.org/html/2408.04290v1">
https://arxiv.org/html/2408.04290v1
</a>
<br><br>
<b>2. Pneumonia detection from enhanced Pneumonia-Chest X-Ray images based on Double SGAN model</b><br>
Zhijing Xu & Haoyang Zhang<br>
<a href="https://www.nature.com/articles/s41598-026-39785-w" >
https://www.nature.com/articles/s41598-026-39785-w
</a>
<br><br>
<b>3. Leveraging human expert image annotations to improve pneumonia 
differentiation through human knowledge distillation</b><br>
Daniel Schaudt, Reinhold von Schwerin, Alexander Hafner, Pascal Riedel, <br>
Christian Späte, Manfred Reichert, Andreas Hinteregger, Meinrad Beer & Christopher Kloth<br>
<a href ="https://www.nature.com/articles/s41598-023-36148-7?fromPaywallRec=false">
https://www.nature.com/articles/s41598-023-36148-7?fromPaywallRec=false
</a>
<br><br>
<b>4. TensorFlow-FlexUNet-Image-Segmentation-Bacterial-Pneumonia-Chest-X-Ray</b><br>
Toshiyuki Arai<br>
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Bacterial-Pneumonia-Chest-X-Ray">
https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Bacterial-Pneumonia-Chest-X-Ray
</a>
<br><br>
<b>5. TensorFlow-FlexUNet-Image-Segmentation-Pneumonia-Chest-X-Ray</b><br>
Toshiyuki Arai<br>
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Pneumonia-Chest-X-Ray">
https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Pneumonia-Chest-X-Ray
</a>
<br><br>
<b>6. TensorFlow-FlexUNet-Image-Segmentation-Chest-X-Ray</b><br>
Toshiyuki Arai<br>
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Chest-X-Ray">
https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Chest-X-Ray
</a>
<br><br>
<b>7. TensrFlow-FlexUNet-Image-Segmentation-Model</b><br>
Toshiyuki Arai<br>
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model">
https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model
</a>
<br><br>
