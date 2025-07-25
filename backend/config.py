import os
from dotenv import load_dotenv

load_dotenv()

MONGO_URL = os.getenv("MONGO_URL")
ROBOFLOW_API_URL = os.getenv("ROBOFLOW_API_URL")
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
MODEL_NAME= os.getenv("MODEL_NAME")
