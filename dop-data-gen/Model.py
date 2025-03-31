'''
The Model.py is used prevent incorrect data typing 
any incoming data will have the correct types and structure
'''


from pydantic import BaseModel

class GPSData(BaseModel):
    latitude: float
    longitude: float
    gdop: float
    pdop: float
    hdop: float
    vdop: float
    tdop: float
    num_in_view: float
    measurement: str




