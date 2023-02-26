# BearingFaultDetection

This code is for Bearing Faul Detection competition in DC platform. Rank 40/1388.

Please see offical website: https://challenge.datacastle.cn/v3/cmptDetail.html?id=248

## Code Overview

1. `data/` directory stores the train data, test data and feature data containing time/frequency/time-frequency feature.
2. `analyze/` directory is used to analyze the correlatoins between features by a correlation matrix.
3. `preprocessing/` directory is main part to preprocess data: 1) Use wavelet filter to denoise data; 2) Use wavelet decoposing tool to extract time-frequency domain feature of dat. 3) And finally, apply PCA to reduce dimension of feature.
4. `RF.py` uses random forest as our classifier to train model.

## Visualization

### Framework
<img width="552" alt="image" src="https://user-images.githubusercontent.com/37136730/156761607-e2b13f4e-1ca4-42c3-a1a6-13bd4cf70b72.png">

### Comparison of signal processed by wavelet filter 
<img width="602" alt="image" src="https://user-images.githubusercontent.com/37136730/156761791-27b16606-f96b-4eb4-b501-b2849eaeb580.png">

### Heatmap of correlation matrix of features
<img width="572" alt="image" src="https://user-images.githubusercontent.com/37136730/156761882-f7512dd3-d436-4ecd-b2fc-fa993c0a014c.png">

## Report 
Please refer to the report.pdf for details.
