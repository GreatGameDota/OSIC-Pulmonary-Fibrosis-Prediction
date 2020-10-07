
# OSIC Pulmonary Fibrosis Progression

My 34th place solution to the [OSIC Pulmonary Fibrosis Progression](https://www.kaggle.com/c/osic-pulmonary-fibrosis-progression/) competition hosted on Kaggle by OSIC.

## Initial Thoughts

This was an interesting competition for me because I saw that everyone else were only using the tabular data. I have no experience at all with tabular data comps and I like CV so I was stubborn and forced myself to use the CT scans. I was discouraged by my poor LB score but I keep pushing and it turned out to be the correct decision!

## Overview

My final solution was an ensemble of 3 5fold Resnet50 models. They were trained on windowed lung ct scan images along with the meta data. I used Google Colab Pro to train all models for 30 epochs.

## Models

Final model was a simple pretrained Resnet50 with a image and meta data part. The image part was simply pretrained model -> pool -> flatten -> dropout -> concat. And the meta model was a simple head with features -> linear -> relu -> linear -> relu -> concat. The final models either had 512->1024 or 100->100 features for the head. And finally a simple linear layer for the 3 FVC output.

## Dataset

For each scan I loaded in all the dicom files then converted them to HU. I then cropped all the images and reshaped them to the size 50x512x512. After doing that I then windowed each image into three parts. For a detailed explanation of the windowing take a look at [this](https://www.kaggle.com/c/rsna-str-pulmonary-embolism-detection/discussion/182930) post (I used the same function and window values as shared there, thanks so much Ian Pan!). After I got the three windowed images I simply saved them as pngs.
Each image looked similar to this:  
![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F3543139%2F3b1f2461da6d31658806e1ace841dd5f%2Flung.png?generation=1602033034958099&alt=media)

I made the dataset I used public here: https://www.kaggle.com/greatgamedota/osic-windowed-lung-images

For meta data I used the same meta data from @ulrich07 's [baseline kernal](https://www.kaggle.com/ulrich07/osic-multiple-quantile-regression-starter) except I only used the base Percent value as it increased my CV to LB to PB correlation.

## Augmentation

- Coarse dropout
- SSR
- Horizontal + Vertical flip
- For one model: Random Saturation and Brightness

No tabular/meta augmentation

## Training

- Adam optimizer with Reduce on Plateau scheduler
- Trained with an LR of .003 for 30 epochs
- Batch size of 16 (bs of 4 for one model)
- Trained using Quantile Regression with .8 qloss + .2 metric loss (same as Ulrich's kernal)
- Trained 5 fold for every model
- Didn't use any batch accumulation or mixed precision
- Saved checkpoint based on best validation score (all weeks)

For training I removed 6 entire patients because their CT scans were broken. I then split each patient into a fold by randomly shuffling them then using GroupKFold.
![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F3543139%2F69a1967fbe3f4067d7fd9f6e5e91350c%2Ffolds.png?generation=1602036568284315&alt=media)

Then while training I pick a random unique patient and randomly select an image from 10-40 (since the first and last 10 images don't contain any lung info). I then made it so that each iteration lasted 4 * amount of patients. For validation I only picked the 15th image since it was the middle image that usually contains the most information.

## Ensembling/Blending

Simple mean average of their FVC predictions and confidence

## Final Submission

My final submission was a blend of 3 Resnet50 models:
![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F3543139%2F773b5cc6fdf39b2fe77ec08d4b295684%2Ffinal%20sub.png?generation=1602040599569002&alt=media)
And another point is that the single model that would have scored gold had the same training parameters as the other models except the added brightness/saturation augmentation

## What didn't work

- 3d resnets (I tried for at least a month with these)
- Linear Decay Regression
- Any other type of model besides Resnets and Efficientnets (determinism issues)
- Efficientnets
- Simple meta data head (just concat)
- Random erase augmentation

## Final Thoughts

I want to say again that this was an awesome comp that I am so glad I participated in! Very glad to get my third medal and second silver medal!

My previous competition: [Melanoma Classification](https://github.com/GreatGameDota/SIIM-ISIC-Melanoma-Classification)

My next competition: [placeholder]
