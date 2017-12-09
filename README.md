## RNN Analysis of Amazon Fine Food Reviews

Notes: 
- seaborn does not exist in the working AMI bitfusion
## Final Project TODO:
- [ ] create report.pdf file
- [ ] source code file, restructure README for submission, add ipynb notebook

## Data preprocessing
TODO:
- [x] Create graphs and detailed information about the data set

https://github.com/yanndupis/RNN-Amazon-Fine-Food-Reviews/blob/master/Amazon%20Fine%20Food%20Reviews.ipynb

- [x] test in train in tflearn
- [x] Look into a bigger beef cake
- [ ] look into the maxlen for each review & graph
- [x] define variable maxlen (siraj used maxlen=100)
- [x] remove pos_tagging (prepositions)
- [x] Train data

## Lexicon Approach
TODO:
- [ ] Create Lexicon based approach to compare against RNN

## Training
TODO:
- [ ] Create 1 model and train 5 different test sets 

## Visualizations
TODO:
- [ ] Receiver Operating Characteristics Curve
- [ ] classification_report 
- [ ] confusion matrix

## Report
Environment setup is done by using a Bitfusion AMI on AWS. This instance, Bitfusion, is an UBuntu 14 AMI that is pre-installed with Nvidia Drivers, Cuda 7.5 Toolkit, cuDNN 5.1, TensorFlow 1.1.0, TFLearn, TensorFlow Serving, TensorFlow TensorBoard, Keras, Magenta, scikit-learn, Python 2 & 3 support, Hyperas, PyCuda, Pandas, NumPy, SciPy, Matplotlib, h5py, Enum34, SymPy, OpenCV and Jupyter to leverage Nvidia GPU as well as CPU instances. 

The AMI can be accessed here at https://aws.amazon.com/marketplace/pp/B01EYKBEQ0/ref=_ptnr_wp_blog_post

The dataset is stored in our S3 cloud storage account, which is already set to be access from the code. 

Result can be directly accessed by running the jupyter notebook file on Bitfusion. The instance we ran the notebook on is (insert Bitfusion AMI tier here)

