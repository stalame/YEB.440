"""
"""

import requests
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env file

API_KEY = os.getenv("FINGRID_API_KEY") # API key here

url = "https://data.fingrid.fi/api/data"

params = {
    "datasets": 124,
    "startTime": "2025-01-01T00:00:00Z",
    "endTime": "2025-01-02T00:00:00Z",
    "pageSize": 1000
}

response = requests.get(
    url,
    params=params,
    headers={"x-api-key": API_KEY}
)

response.raise_for_status()

data = response.json()["data"]

df = pd.DataFrame(data)

# Convert timestamps to Finland time
df["startTime"] = pd.to_datetime(df["startTime"], utc=True)
df["time_fi"] = df["startTime"].dt.tz_convert("Europe/Helsinki")

# Plot
plt.figure(figsize=(12, 6))
plt.plot(df["time_fi"], df["value"], marker="o")

plt.title("Finnish Electricity Consumption\n1 January 2025")
plt.xlabel("Time")
plt.ylabel("Consumption (MW)")
plt.grid(True)

plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    r"./img/electricity_consumption_2025_01_01.png",
    dpi=300,
    bbox_inches="tight"
)

print("Saved!")

plt.show()