# SageMaker
- Machine Learning is the hottest topic in the current era and the leading cloud provider Amazon web service (AWS) provides lots of tools to explore Machine Learning, creating models with a high accuracy rate.
- makes you familiar with one of those services on AWS i.e. Amazon SageMaker which helps in creating efficient and more accuracy rate Machine learning models and the other benefit is that you can use other AWS services in your model such as S3 bucket, amazon Lambda for monitoring the performance of your ML model you can use AWS CloudWatch which is a monitoring tool. 
# Pytorch
- PyTorch is an open source machine learning library for Python and is completely based on Torch. 
- It is primarily used for applications such as natural language processing
- PyTorch redesigns and implements Torch in Python while sharing the same core C libraries for the backend code.
- PyTorch developers tuned this back-end code to run Python efficiently. They also kept the GPU based hardware acceleration as well as the extensibility features that made Lua-based Torch.

# Text Generator
- Text Generator by building Recurrent Long Short Term Memory Network. The conceptual procedure of training the network is to first feed the network a mapping of each character present in the text on which the network is training to a unique number.
- Which the network studies the mapping and predict the next word

# LSTM
- LSTM networks are an extension of recurrent neural networks (RNNs) mainly introduced to handle situations where RNNs fail. 
Talking about RNN, it is a network that works on the present input by taking into consideration the previous output (feedback) and storing in its memory for a short period of time (short-term memory).
- Thus an LSTM recurrent unit tries to “remember” all the past knowledge that the network is seen so far and to “forget” irrelevant data. This is done by introducing different activation function layers called “gates” for different purposes.
Each LSTM recurrent unit also maintains a vector called the Internal Cell State which conceptually describes the information that was chosen to be retained by the previous LSTM recurrent unit. 
- The LSTM does have the ability to remove or add information to the cell state, carefully regulated by structures called gates  

## Code Description


    File Name : Engine.py
    File Description : Main class for starting the model training lifecycle

    File Name : Model.py
    File Description : Class of LSTM structure
    
    File Name : Train.py
    File Description : Class for starting the model training 
    
    File Name : Predict.py
    File Description : Class for evaluvating the model
    
    File Name : Tokens.py
    File Description : Class for Generatin Tokens and Save it for Testing and Training

    File Name : Data_Processing.py
    File Description : Code to load and transform the dataset. 
    
## Steps to Run

There are two ways to execute the end to end flow.

- Modular Code
- IPython

### Modular code

- Create virtualenv
- Install requirements `pip install -r requirements.txt`
- Run Code `python Engine.py`
- Check output for all the visualization

### IPython Google Colab

Follow the instructions in the notebook `Sage_Maker_Text_Generator.ipynb`

