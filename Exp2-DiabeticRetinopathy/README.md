
</br>
</br>

## For Type I and Type II:

1- download data from challenge website: </br>
Only train folder </br>
https://www.kaggle.com/c/aptos2019-blindness-detection

2- resize all images to 256x256x3 and place them in a folder named "resizedversion_256x256" </br>
3- Place the folder of 2, in _data_ subdirectory </br>
Indeed the data subdirectory should contains _resizedversion_256x256_ folder and _train.csv_

4- execute the scripts </br>
for Type I: </br>
> python  Pr_Tensorization_v0.py
 
for Type II:</br> 
> python  Pr_Tensorization_v1_color.py

It will produce the results in a folder named "1"

## For Type III:

1- download the dataset: </br>
https://github.com/deepdrdoc/Deep-Diabetic-Retinopathy-Image-Dataset-DeepDRiD- </br>
only subdirectory _/regular_fundus_images/regular-fundus-training/_ is required. </br>
Indeed: https://github.com/deepdrdoc/Deep-Diabetic-Retinopathy-Image-Dataset-DeepDRiD-/tree/master/regular_fundus_images/regular-fundus-training 

2- resize all images to 256x256x3 and place them in a folder named "resizedversion_256x256_drid" </br> 
You can use my converter code with MATLAB: _Pr_Resize.m_ . make sure to set the directory path. </br> 

3- Place the folder of 2, in _data_ subdirectory </br>
Indeed the data subdirectory should contains _resizedversion_256x256_drid_ folder and _data.csv_
4- execute the script: </br>
> python Pr_Tensorization_v4_color.py

It will produce the results in a folder named "v4"