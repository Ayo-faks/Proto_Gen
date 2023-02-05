# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"
from __future__ import print_function

from rasa_sdk.events import AllSlotsReset
from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher



import datetime
from datetime import datetime, timedelta
import os.path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import pickle




class ActionHelloWorld(Action):

    def name(self) -> Text:
        return "action_hello_world"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        dispatcher.utter_message(text="Hello World!")

        return []


class AddEventToCalendar(Action):

    def name(self) -> Text:
        return "action_add_event"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        event_name = tracker.get_slot('event')
        time = tracker.get_slot('time')
        new_time = datetime.strptime(time, '%d/%m/%y %H:%M:%S')

        add_event(event_name, new_time)

        dispatcher.utter_message(text="Event Added")

        return [AllSlotsReset()]

class getEvent(Action):

    def name(self) -> Text:
        return "action_get_event"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        event_name = get_event()

        print(event_name)
        #confirmed_event = tracker.get_slot(Any)
        dispatcher.utter_message(text="got events {name}".format(name= event_name))
        return []

# If modifying these scopes, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/calendar']

CREDENTIALS_FILE = 'credentials.json'

def get_calendar_service():
   creds = None
   # The file token.pickle stores the user's access and refresh tokens, and is
   # created automatically when the authorization flow completes for the first
   # time.
   if os.path.exists('token.pickle'):
       with open('token.pickle', 'rb') as token:
           creds = pickle.load(token)
   # If there are no (valid) credentials available, let the user log in.
   if not creds or not creds.valid:
       if creds and creds.expired and creds.refresh_token:
           creds.refresh(Request())
       else:
           flow = InstalledAppFlow.from_client_secrets_file(
               CREDENTIALS_FILE, SCOPES)
           creds = flow.run_local_server(port=0)

       # Save the credentials for the next run
       with open('token.pickle', 'wb') as token:
           pickle.dump(creds, token)

   service = build('calendar', 'v3', credentials=creds)
   return service

def add_event(event_name, time):
   # creates one hour event tomorrow 10 AM IST
   service = get_calendar_service()

    

#    d = datetime.now().date()
#    tomorrow = datetime(d.year, d.month, d.day, 10)+timedelta(days=1)
#    start = tomorrow.isoformat()
   end = (time + timedelta(hours=1)).isoformat()



   event_result = service.events().insert(calendarId='primary',
       body={
           "summary": event_name,
           "description": 'This is a tutorial example of automating google calendar with python',
           "start": {"dateTime": time.isoformat(), "timeZone": 'Asia/Kolkata'},
           "end": {"dateTime": end, "timeZone": 'Asia/Kolkata'},
       }
   ).execute()

   print("created event")
   print("id: ", event_result['id'])
   print("summary: ", event_result['summary'])
   print("starts at: ", event_result['start']['dateTime'])
   print("ends at: ", event_result['end']['dateTime'])


def get_event():

    service = get_calendar_service() 
    now = datetime.utcnow().isoformat() + 'Z'
    events = service.events().list( calendarId='primary', timeMin=now,
       maxResults=10, singleEvents=True,
       orderBy='startTime').execute().get("items",[])

    print(events[0]["summary"])
    return events[0]["summary"]

def do_event():

    service = get_calendar_service() 
    now = datetime.utcnow().isoformat() + 'Z'
    events = service.events().list( calendarId='primary', timeMin=now,
       maxResults=10, singleEvents=True,
       orderBy='startTime').execute().get("items",[])

    print(events[0]["end"])
    return events[0]["end"]

class ActionDoEvent(Action):

    def name(self) -> Text:
        return "action_do_event"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        event_name = do_event()

        print(event_name)
        #confirmed_event = tracker.get_slot(Any)
        dispatcher.utter_message(text="got events {name}".format(name= event_name))
        return []
        
        
import pinecone
import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import BartTokenizer, BartForConditionalGeneration

min_length = 200
max_length = 400


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

def query_pinecone(query, top_k):
    # generate embeddings for the query
    xq = retriever.encode([query]).tolist()
    # search pinecone index for context passage with the answer
    xc = index.query(xq, top_k=top_k, include_metadata=True)
    return xc
  
def format_query(query, context):
    context = [f"<P> {m['metadata']['passage_text']}" for m in context]
    context = " ".join(context)
    query = f"question: {query} context: {context}"
    return query
    

class ActionGenAnswer(Action):

    def name(self) -> Text:
        return "action_hello_world"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        query = tracker.latest_message['text']
        
        if query != "":
            
           
            xq = retriever.encode([query]).tolist()
            xc = index.query(xq, top_k=1, include_metadata=True)
            query = format_query(query, xc["matches"]) 
        
            # genrate answer from LLM
            answer = generator.generate(query, min_length=min_length, max_length=max_length)
        
           
        dispatcher.utter_message(text=answer)
         


# get last user message

        
    



   
 