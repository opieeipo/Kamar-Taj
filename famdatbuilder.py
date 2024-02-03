import os
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from datetime import datetime
from langchain.text_splitter import TokenTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from tqdm import tqdm
from langchain_community.llms import OpenAI
import argparse

parser=argparse.ArgumentParser()

parser.add_argument("--apikey", help="Provide the OpenAI APIKey")
parser.add_argument("--output", help="Provide the outputpath for the faiss index")
parser.add_argument("--texts", help="Provide the location of the fam,fah,dssrtexts")

args=parser.parse_args()

#Set the variables from the command line arguments
apikey=args.apikey
output=args.output
td=args.texts

if apikey is None or output is None or td is None:
 parser.print_help()
else:  
  text_splitter = TokenTextSplitter.from_tiktoken_encoder(model_name="gpt-3.5-turbo",encoding_name="cl100k_base")
  oe = OpenAIEmbeddings(openai_api_key=apikey)

  def current_time():
   now = datetime.now()
   return now.strftime("%H:%M:%S")
    
  def loadAllDocuments(pd,od):
   files = os.listdir(pd)
   spath = od
   
   print (f"{current_time()}: Checking for existing database at {od}")   
   if (os.path.exists(spath)):    
    print (f"{current_time()}: Found and loading {od}")   
    db = FAISS.load_local(spath,oe)
   else:
    print (f"{current_time()}: No database found will create after first integration")   
    db=None
   print (f"{current_time()}: Loading text files into database from {pd}")   
   pbar = tqdm(files, unit='file')
   for file in pbar:
     #initialize the documents array
     langchain_documents=[]    
     #create the path where the individual file will be read from
     fp=os.path.join(pd,file)
     #load in the file using the textloader
     loader = TextLoader(fp,encoding='utf8')     
     #populate the documents array with the loader
     documents=loader.load()
     #This adds the documents just loaded into the array we did a few lines back
     langchain_documents.extend(documents)
     #split the documents into chunks using chat-gpt
     chunks = text_splitter.split_documents(langchain_documents)          
     #update the description on the progress bar
     pbar.set_description(f"{current_time()}: Processing {file}(integrating)")     
     if db is None:      
      pbar.set_description(f"{current_time()}: Creating Database @ {od} ")
      db = FAISS.from_documents(chunks,embedding=oe)
      pbar.set_description(f"{current_time()}: Processing {file}(saving DB) to {od}")                    
     else:    
       db.add_documents(chunks)
       pbar.set_description(f"{current_time()}: Processing {file}(saving DB) to {od}")      
     
     db.save_local(spath)
  
  loadAllDocuments(td,output)



