# CTPN - Text Detection with tensorflow 2.10

## Requirements
* Python 3.10.0
* cuDNN 8.7
* CUDA 11.8
* Tensorflow 2.10
## VOC2007 Dataset
* Download 
* https://drive.google.com/drive/folders/11RSNqPT8zTaMiZVTcPcwAFswLvAzZXmo?usp=sharing
## Usages

### Prediction
* 預測 <br>
`python ctpn_predict.py --image_path {Your Path}\test.png 
--output_file_path {Your Path}\output.png 
--weights_file_path {Your Path}\CTPN_TF210\text_detection_ctpn\weights\weights-ctpnlstm-01.hdf5`

### Training (Strat From Scratch)
* 訓練(從頭開始) <br>
`python ctpn_train.py 
--anno_dir {Your Path}\VOCdevkit\VOC2007\Annotations
--images_dir   {Your Path}\JPEGImages
--epochs {number of epochs} (Greater than 10)`

### Resume Training (Weight File Assigned)
* 恢復訓練 (指定權重檔) <br>
`python ctpn_train.py
--anno_dir {Your Path}\VOCdevkit\VOC2007\Annotations
--images_dir    {Your Path}\VOCdevkit\VOC2007\JPEGImages
--weights_file_path {Your Path}\CTPN_TF210\text_detection_ctpn\weights\weights-ctpnlstm-01.hdf5
--epochs 11`
