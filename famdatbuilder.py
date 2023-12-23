import os
import openai
import shutil

from langchain import *
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import Document
from datetime import datetime
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from bs4 import BeautifulSoup 
from tqdm import tqdm
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
import sys
import argparse


parser=argparse.ArgumentParser()

parser.add_argument("--apikey", help="Provide the OpenAI APIKey (i.e. sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx) ")
parser.add_argument("--output", help="Provide the outputpath for the faiss index")
parser.add_argument("--texts", help="Provide the location of the fam,fah,dssrtexts")

args=parser.parse_args()

#Set the variables from the command line arguments
apikey=args.apikey
output=args.output
td=args.texts
OPENAI_API_KEY=apikey

if apikey is None or output is None or td is None:
 parser.print_help()
else:
 
  text_splitter = RecursiveCharacterTextSplitter(
     chunk_size=1000,
     chunk_overlap=100,
     length_function=len,
  )
  model_name = "sentence-transformers/all-mpnet-base-v2"
  model_kwargs = {"device": "cuda"}

  hf = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

  def current_time():
   now = datetime.now()
   return now.strftime("%H:%M:%S")

  def convert_to_txt(sd,nm): 
   files=os.listdir(sd) 

   for file in files:   
    html = sd+"/"+file
    tfile = file.split(".")[0]
    parent = f"{nm}/{tfile}.txt"
    if (os.path.exists(parent)):
     os.remove(parent)
    f = open (parent,"w",encoding="utf-8")   
    r = open (sd+"/"+file,"r",encoding="utf-8")
    soup = BeautifulSoup(r.read())
    t= soup.get_text('\n').replace("\n"," ")
    print (f"Writing to {parent} from {html}")
    f.writelines(t)
    r.close()
    f.close()

  def loadAllDocuments(pd):
   files = os.listdir(pd)
 
   spath = os.path.join(pd,"fvdb","fam.faiss")
 
   if (os.path.exists(spath)):
     db = FAISS.load_local(spath,hf)
   else:
    db=None
   print (f"{current_time()}: Loading text files into database from {pd}")
   pbar = tqdm(files)
   for file in pbar:
     langchain_documents=[]    
     pbar.set_description(f"{current_time()}:  Processing {file}(loading)(1/5)")
     loader = TextLoader(f"{pd}/{file}")     
     documents=loader.load()
     pbar.set_description(f"{current_time()}:  Processing {file}(prepping)(2/5)")
     langchain_documents.extend(documents)
     pbar.set_description(f"{current_time()}:  Processing {file}(chunking)(3/5)")
     chunks = text_splitter.split_documents(langchain_documents)
     pbar.set_description(f"{current_time()}:  Processing {file}(integrating)(4/5)")
     if db is None:
       db = FAISS.from_documents(chunks,embedding=hf)
     else:
       db.add_documents(chunks)
     pbar.set_description(f"{current_time()}:  Processing {file}(saving DB)(5/5)")
     db.save_local(spath)
     
    
 
  def load_documents(pd):
   dirs = os.listdir(pd)
   embeddings = OpenAIEmbeddings()
   for dir in dirs:
    basedir = os.path.join(pd,dir)
    print (basedir)
    spath=""
    db = None
    if (os.path.isdir(pd+"/"+dir)):   
     files = os.listdir(basedir)   
     langchain_documents=[]
     for file in tqdm(files):    
      try:     
       loader = TextLoader(f"{basedir}/{file}")     
       documents=loader.load()
       langchain_documents.extend(documents)        
      except Exception:
       continue
     print(f"\n{current_time()}: Chunking the documents for {dir}")
     chunks = text_splitter.split_documents(langchain_documents)
     print(f"\n{current_time()}: Integrating chunks into the vector DB")
     db = FAISS.from_documents(chunks,embedding=hf)
     print(f"\n{current_time()}:  saving db to {spath}")
     spath = pd+"/../vdb/"+dir+".faiss"
     db.save_local(spath)
   return 0

  def merge_faiss(id,od):
   files = os.listdir(id)
   size = len(files)
   odn = od+"/master.faiss" 
  
   shutil.copytree(id+files[0], odn)
   embeddings = OpenAIEmbeddings()
   odb= FAISS.load_local(odn,embeddings) 
   i=1 
   while i < size:
    idb = FAISS.load_local(id+files[i],embeddings)  
    odb.merge_from(idb)
    i=i+1
 

  def get_dir(pd):
   dirs = os.listdir(pd)
   for dir in dirs:
     pdname = f"./texts/"+dir.split("/")[len(dir.split("/"))-1]
     os.mkdir(pdname)
     if (os.path.isdir(pd+"/"+dir)):
      convert_to_txt(pd+"/"+dir,pdname) 


  #load_documents(td)
  print(td)


