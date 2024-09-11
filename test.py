from sentence_transformers import SentenceTransformer
import pandas as pd
import matplotlib.pyplot as plt
import torch
import my_net as mn
import my_sampler as ms

#------------------------------------------
model_file = "/home/mkr/lama3/model-L8-500.pth"
data_file = '/home/mkr/lama3/data/test.csv'
output_file = '/home/mkr/lama3/data/predicted.csv'
classes = [0, 1]
   
def append_row(df, row):
    return pd.concat([
                df, 
                pd.DataFrame([row], columns=row.index)]
           ).reset_index(drop=True)

#---------------------------------------------------------------------
# Let's go!
#---------------------------------------------------------------------

input_data = pd.read_csv(data_file, keep_default_na=False)
output_data = pd.DataFrame(columns=['id', 'target'])

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

#init the model
model = mn.nlp_nn()
checkpoint = torch.load(model_file, map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)

model.eval()

emb_model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1", trust_remote_code=True)
    
#compute predictions for test data
for idx in range(input_data.shape[0]):
    id = input_data.iloc[idx, 0]
    keyword = input_data.iloc[idx, 1]
    location = input_data.iloc[idx, 2]
    text = input_data.iloc[idx, 3]
        
    X = torch.cat((torch.from_numpy(emb_model.encode(keyword)), torch.from_numpy(emb_model.encode(text))), dim=0)


    X = X.to(device)

    with torch.no_grad():
        outputs = model(X)
    
    _, pred_idx = torch.max(outputs.data, dim=0)
    
    target = classes[pred_idx]

    new_row = pd.Series({'id': id, 'target': target})
    output_data = append_row(output_data, new_row)

output_data.to_csv(output_file, index=False) 
