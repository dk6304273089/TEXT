import streamlit as st
import translators as ts
from transformers import PegasusTokenizer,PegasusForConditionalGeneration
import torch
import pandas as pd

path = "./model/lstmmodelgpu.pth"

h=st.sidebar.selectbox("Select Activity",["Translator and summarizer","Pegasus Tokenizer"])

if h=="Translator and Summarizer" :
    st.title("Translate and summarize")
    f=st.sidebar.text_area("Paste Text Here",height=400)
    button1=st.sidebar.button("Predict")
    if button1:
        st.session_state.load_state=True
        col1,col2=st.columns(2)
        c=ts.google(f)
        tokenizer = PegasusTokenizer.from_pretrained("./model/")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = torch.load(path)
        batch = tokenizer(c, truncation=True, padding='longest', return_tensors="pt").to(device)
        translated = model.generate(**batch)
        tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
        col1_expander=col1.expander("Expand Translated Text")
        with col1_expander:
            col1_expander.write(c)

        col2_expander=col2.expander("Expand Summarized Text")
        with col2_expander:
            col2_expander.write(tgt_text[0])

else:
    st.title("Pegasus Tokenizer")
    user_input=st.text_area("Enter the Text")
    button1 = st.button("Predict")
    if button1:
        tokenizer2 = PegasusTokenizer.from_pretrained("./model/")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        inputs = tokenizer2(user_input, truncation=True, padding='longest', return_tensors="pt").to(device)
        j = pd.DataFrame()
        k = []
        for g in inputs["input_ids"][0]:
            k.append(tokenizer2.decode(g, skip_special_tokens=False, clean_up_tokenization_spaces=False))
        j["Text"]=k
        j["Tensor"] = inputs["input_ids"][0]
        st.write(j)



