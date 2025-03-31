from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from influxdb_client import InfluxDBClient
from pydantic import BaseModel
from typing import List

# InfluxDB connection settings
INFLUXDB_HOST = "localhost"
INFLUXDB_PORT = "8086"
INFLUXDB_USERNAME = "username"
INFLUXDB_PASSWORD = "password"
INFLUXDB_DATABASE = "mydopdb"
INFLUX_URL = f"http://{INFLUXDB_HOST}:{INFLUXDB_PORT}"

# Initialize FastAPI app
app = FastAPI()

# Configure CORS (Cross-Origin Resource Sharing)
origins = [
    "http://localhost:8080",  # Vue.js default port
    "http://localhost:5174",  # Vite default port (if using Vite)
    "http://127.0.0.1:8080",
    "http://127.0.0.1:5174",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allow these origins to make requests
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Create InfluxDB client
client = InfluxDBClient(url=INFLUX_URL, username=INFLUXDB_USERNAME, password=INFLUXDB_PASSWORD, database=INFLUXDB_DATABASE)

# Define the DopData model to structure the data
class DopData(BaseModel):
    latitude: float
    longitude: float
    pdop: float

@app.get("/dop-data", response_model=List[DopData])
async def get_dop_data():
    query = f'''
    SELECT latitude, longitude, pdop
    FROM "dop_data"  -- Replace with your InfluxDB measurement name
    WHERE time > now() - 1d  -- Last 24 hours of data
    '''
    
    # Execute query to InfluxDB
    result = client.query(query)
    
    # Parse the result and return it as a list of DopData
    data = []
    for table in result.get_points():
        data.append({
            "latitude": table['latitude'],
            "longitude": table['longitude'],
            "pdop": table['pdop'],
        })
    
    return data
