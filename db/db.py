import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from db.models import Base

# Read all key=value pairs from the .env file and load them into environment variables
load_dotenv()

DATABASE_URL = os.getenv('DATABASE_URL')
if not DATABASE_URL:
    raise ValueError('DATABASE_URL is not set')

#Create an SQAlchemy engine -  this handles the connection to the PostgreSQL database
engine = create_engine(DATABASE_URL)

#This is a factory for creating sessions. This is how we interact with the DB (Add,delete,update...)
SessionLocal = sessionmaker(bind = engine)

#This will create the tables defined in models.py
def init_db():
    Base.metadata.create_all(bind = engine)