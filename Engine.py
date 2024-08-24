import boto3
## Connecting to S3 Bucket
bucket = 'textgenerationbucket'
key = 'text_data/Review_test.csv'
s3_client = boto3.client('s3',
                         aws_access_key_id='',
                         aws_secret_access_key='',
                         region_name=''
                         )
obj = s3_client.get_object(Bucket=bucket, Key=key)

# # Importing Nessesary Packages:
import numpy as np
import torch as torch

# Importing the Nessesary Files:
from MLPipeline.Data_Processing import Data_Processing
from MLPipeline.Tokens import Tokens
from MLPipeline.Model import Model
from MLPipeline.Train import Train
from MLPipeline.Predict import Predict

f1 = Data_Processing()
f2 = Tokens()
f3 = Train()
f4 = Predict()

# Loading Data From S3:
data = f1.load_data(obj)

# Taking the Text Column:
text = data["Text"].values
seq = []
for i in range(100):
    seqi = f1.create_seq(text[i])
    seq.extend([s for s in seqi if len(s.split(" ")) == 11])
# print(seq[1])

# Splitting the Data into X and Y
x, y = f1.splitting(seq)

# Generating the Tokens:
int2token, token2int = f2.tokens(seq)

# Vocab Size:
vocab_size = len(int2token)
print("Vocab_Size:", vocab_size)

# convert text sequences to integer sequences
x_int = [f2.get_integer_seq(i) for i in x]
y_int = [f2.get_integer_seq(i) for i in x]

# convert lists to numpy arrays
x_int1 = torch.tensor(np.array(x_int))
y_int1 = torch.tensor(np.array(y_int))

# instantiate the model
net = Model(vocab_size)
print(net)

# push the model to GPU (avoid it if you are not using the GPU)
# net.cuda()

# Training The Model:
f3.train(x_int1, y_int1, net, epochs=1)

output_path = 'C:\Project\Text_Generator_Sagemaker\Output'
PATH = output_path + "\model.pt"
torch.save(net.state_dict(), PATH)

print("Predicted OutPut: ")
print(f4.sample(net, 5, prime="amazing product"))
