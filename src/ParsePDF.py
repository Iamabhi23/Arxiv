# Written By Nathan Rigoni for the Fall Capstone of the DAAN 888

import pandas as pd
from io import StringIO
import os
import re
from gensim.models.doc2vec import TaggedDocument
from gensim.parsing.preprocessing import STOPWORDS
from gensim import corpora
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import remove_stopwords

from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage

class PDF_loader(object):
    """
    This calss acts as a generator object to load the pdfs one by 
    one for conversion into a line sentence file. The dataset is 
    somewhere on the order of 5gb and so cannot be loaded all at once.
    """
    def __init__(self, path:str):
        self.path = path
        self.files = os.listdir(path)

    def __iter__(self):
        for file in self.files:                        
            yield simple_preprocess(_pdf_text_reader(self.path+file))

def _pdf_text_reader(pdf_file_name, pages=None):
    if pages:
        pagenums = set(pages)
    else:
        pagenums = set()

    ## 1) Initiate the Pdf text converter and interpreter
    textOutput = StringIO()
    manager = PDFResourceManager()
    converter = TextConverter(manager, textOutput, laparams=LAParams())
    interpreter = PDFPageInterpreter(manager, converter)

    ## 2) Extract text from file using the interpreter
    bad_files = []
    infile = open(pdf_file_name, 'rb')
    try:
        pages = PDFPage.get_pages(infile, pagenums)
        for page in pages:
            interpreter.process_page(page)   
    except:
        print(pdf_file_name)
        bad_files.append(pdf_file_name)    
    infile.close()
    
    ## 3) Extract the paragraphs and close the connections
    paras = textOutput.getvalue()   
    converter.close()
    textOutput.close
    paras = _string_clean(paras, keep_nums=False)
    return paras

def _string_clean(text, keep_nums=True):
    if keep_nums is True:
        text = re.sub(r'\S*@\S*\s?', '', text, flags=re.MULTILINE) # remove email
        text = re.sub(r'http\S+', '', text, flags=re.MULTILINE) # remove web addresses
        text = re.sub("\'", "", text) # remove single quotes
        text = re.sub(r'\d\n[0-9]?[a-zA-Z]?\.', '', text)
    else:
        text = re.sub(r'[^A-Za-z0-9 ]+', ' ', text)
        text = re.sub(r'\d+', ' ', text)
        text = re.sub(r'\(\S*\)', ' ', text)
        text = re.sub(r'nnnnnn', ' ', text)
        text = re.sub(r'\n\]\[', ' ', text)
        text = re.sub(r'\S*@\S*\s?', ' ', text, flags=re.MULTILINE) # remove email
        text = re.sub(r'http\S+', ' ', text, flags=re.MULTILINE) # remove web addresses
        text = re.sub("\'", " ", text) # remove single quotes
        
    text = " ".join([word.lower() for word in text.split() if len(word)>1 and word not in STOPWORDS])
    return text