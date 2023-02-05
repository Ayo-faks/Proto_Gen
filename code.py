import pinecone
import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import BartTokenizer, BartForConditionalGeneration


class BartGenerator:
    def __init__(self, model_name):
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.generator = BartForConditionalGeneration.from_pretrained(model_name)

    def tokenize(self, query, max_length=1024):
        inputs = self.tokenizer([query], max_length=max_length, return_tensors="pt")
        return inputs

    def generate(self, query, min_length=20, max_length=40):
        inputs = self.tokenize(query)
        ids = self.generator.generate(inputs["input_ids"], num_beams=1, min_length=int(min_length), max_length=int(max_length))
        answer = self.tokenizer.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return answer
    
def init_models():
    retriever = SentenceTransformer("flax-sentence-embeddings/all_datasets_v3_mpnet-base")  
    generator = BartGenerator("vblagoje/bart_lfqa")
    return retriever, generator

PINECONE_KEY = "84cef4db-2780-44fa-ac4f-e4c68e333546"

def init_pinecone():
    pinecone.init(api_key=PINECONE_KEY, environment="us-west1-gcp")  # get a free api key from app.pinecone.io
    return pinecone.Index("abstractive-question-answering")

retriever, generator = init_models()
index = init_pinecone()

def display_answer(answer):

def display_context(title, context, url):
  
  
def format_query(query, context):
    context = [f"<P> {m['metadata']['passage_text']}" for m in context]
    context = " ".join(context)
    query = f"question: {query} context: {context}"
    return query
    

# get last user message
query = tracker.latest_message['text']

if query != "":
    
   
    xq = retriever.encode([query]).tolist()
    xc = index.query(xq, top_k=int(top_k), include_metadata=True)
    query = format_query(query, xc["matches"])

    # genrate answer from LLM
    answer = generator.generate(query, min_length=min_length, max_length=max_length)

   
dispatcher.utter_message(text=answer)
   
for m in xc["matches"]:
    title = m["metadata"]["article_title"]
    url = "https://en.wikipedia.org/wiki/" + title.replace(" ", "_")
    context = m["metadata"]["passage_text"]
    display_context(title, context, url)