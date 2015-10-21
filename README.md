# kaggle-rossmann competition
Prediction model for Kaggle/Rossmann competition.
Contact me on my linkedin if you need more information.

You can find out more about this competition at:
https://www.kaggle.com/c/rossmann-store-sales

https://www.kaggle.com/

# steps
0. clone repo
1. install python, pandas, numpy, scipy/scikit, keras, hdf5, ...
2. run the ./load_data_hdf5.py in src/data directory
3. run the model from models directory for training
4. If you want to experiment with lstm then you need to edit and run create_data.py. You should edit this file and file lstm_simple.py and specify how long sequences should you want to use. This model is not complete, i don't decide yet how to make predictions for test. I welcome pull requests.

# folders
/data                      : datasets

/src/data                  : handling dataset, missing values mostly with pandas

/src/models                : models for prediction

/src/models/results        : results
