

1- download the dataset: </br>
https://github.com/deepdrdoc/Deep-Diabetic-Retinopathy-Image-Dataset-DeepDRiD- </br>
only subdirectory _/regular_fundus_images/regular-fundus-training/_ is required. </br>
Indeed: https://github.com/deepdrdoc/Deep-Diabetic-Retinopathy-Image-Dataset-DeepDRiD-/tree/master/regular_fundus_images/regular-fundus-training 

2- resize all images to 256x256x3 and place them in a folder named "resizedversion_256x256_drid" </br> 
You can use my converter code with MATLAB: _Pr_Resize.m_ . make sure to set the directory path. </br> 

3- Place the folder of 2, in _data_ subdirectory </br>
Indeed the data subdirectory should contains _resizedversion_256x256_drid_ folder and _data.csv_
4- execute the script: </br>
> python Pr_Tensorization_v4_color_DeepEnsemble.py

It will produce the results in a folder named "v4_DE"

<p align="center">
<img src="fig_2.png" alt="drawing" width=80%/>
</p>


For more results and case studies, consider ReadME_crossvalidation.
