import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../src')
from arsearch import ArchiveSearchAbstracts

# Load the model for the app 
AS = ArchiveSearchAbstracts(path = '../models/abstracts') # this is where you swtich between the papers model and the abstract model.

# initialize the streamlit page
st.set_page_config('Arxiv Search', layout='wide')

# add a search text box for users to enter search terms
text = st.text_input("Search text here", )
#check the contents of the box and if it is not empty then use its contents to search the model and return the most similar results
if len(text)>0:
    results = AS.search(text, top_n=5)
    st.dataframe(results.loc[:, ["id", "abstract", "similarity"]].astype(str))

# create a button that the user can use to show a plot of the latent space vectors from the umap reduction.
if st.button('Show topics in latent space'):
    fig = plt.figure(figsize = (10, 10)) # init the figure for th plot
    ax = fig.add_subplot(111) # add the subplot
    ax.scatter(AS.umap_model.embedding_[:, 0], AS.umap_model.embedding_[:, 1], c=AS.cluster.labels_, s=2) # plot the data
    st.pyplot(fig) # push the plot to streamlit for visualization

# create a selection box for the user to select a topic to display a wordloud for that topic
selection = st.selectbox("Select a Topic for wordcloud",
                        np.unique(AS.cluster.labels_))
st.pyplot(AS.get_topic_wordcloud(selection)) # push the wordcloud plot to the streamlit page for visualization
