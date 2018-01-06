## RNN Analysis of Amazon Fine Food Reviews

## Library Dependencies: 
- TFLearn
- NumPy
- SciPy
- Pandas
- Matplotlib
- String
- Sklearn

## Report
Environment setup is done by using a Bitfusion AMI on AWS. This instance, Bitfusion, is an UBuntu 14 AMI that is pre-installed with Nvidia Drivers, Cuda 7.5 Toolkit, cuDNN 5.1, TensorFlow 1.1.0, TFLearn, TensorFlow Serving, TensorFlow TensorBoard, Keras, Magenta, scikit-learn, Python 2 & 3 support, Hyperas, PyCuda, Pandas, NumPy, SciPy, Matplotlib, h5py, Enum34, SymPy, OpenCV and Jupyter to leverage Nvidia GPU as well as CPU instances. 

The AMI can be accessed here at https://aws.amazon.com/marketplace/pp/B01EYKBEQ0/ref=_ptnr_wp_blog_post

The dataset is stored in our S3 cloud storage account, which is already set to be access from the code. 

Result can be directly accessed by running the jupyter notebook file on Bitfusion. The instance we ran the notebook on is r4.4xlarge. You can easily recreate our results by creating the AMI and running the jupyter notebook file on the AMI. 

