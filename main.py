import streamlit as st
from simpletransformers.ner import NERModel
import json 
import pandas as pd

st.title("Named Entity Recognition")

st.write("""
# Explore Us!
Which model is the best?
""")

model_name = st.sidebar.selectbox("Select Model",("bert","roberta"))
st.write(model_name)

sentence = st.text_input("Sentence")
st.write(sentence)

def predict(sentence):
    if sentence :
        model1 = NERModel('bert', 'NERMODEL1',
                  labels=["B-sector","I-sector","B-funda","O","operator","threshold","Join","B-attr","I-funda","TPQty","TPUnit","Sortby", "B-eco","I-eco","B-index","Capitalization","I-","funda","B-security",'I-security','Number','Sector','TPMonth','TPYr','TPRef'],
                  args={"save_eval_checkpoints": False,
        "save_steps": -1,
        "output_dir": "NERMODEL",
        'overwrite_output_dir': True,
        "save_model_every_epoch": False,
        'reprocess_input_data': True, 
        "train_batch_size": 10,'num_train_epochs': 15,"max_seq_length": 64}, use_cuda=False)

        predictions, raw_outputs = model1.predict([sentence])     
        result = json.dumps(predictions[0])
        return result

if sentence :
    result= predict(sentence)
    result=pd.DataFrame(result)
    st.write(result)