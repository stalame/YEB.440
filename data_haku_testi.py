"""
YEB440 Voimalaitokset ja kytketyt energiajärjestelmät
Tietokoneharjoitus 2: Datan haku Fingrid Open Data -rajapinnasta TESTI
Amelia Stalter, 151193060
Kokeillaan, toimiiko Fingrid Open Data yhteys API-avaimella
"""
import requests
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env file

API_KEY = os.getenv("FINGRID_API_KEY") # API key here

headers = {
    "x-api-key": API_KEY
}

# Esimerkki: Fingrid-dataset 124 (Suomen sähköntuotanto)
url = "https://data.fingrid.fi/api/data?datasets=124&pageSize=1"

response = requests.get(url, headers=headers)

if response.status_code == 200:
    print("Yhteys onnistui!")

else:
    print("Yhteys epäonnistui. Status code:", response.status_code)