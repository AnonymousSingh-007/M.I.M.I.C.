import time
import win32api
import csv
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler('app.log'), logging.StreamHandler()])

# Prompt for inputs
user = input("Enter user number (1-4): ")
session = input("Enter session number: ")
duration = int(input("Enter session duration in seconds (e.g., 420 for 7 minutes): "))

filename = f"user{user}_session{session}.csv"
logging.info(f"Starting data collection for {filename}, duration: {duration}s")

data = []
start_time = time.perf_counter()

try:
    while (time.perf_counter() - start_time) < duration:
        current_time = time.perf_counter() - start_time
        x, y = win32api.GetCursorPos()
        data.append((current_time, x, y))
        time.sleep(0.008)  # ~125 Hz
except Exception as e:
    logging.error(f"Error during collection: {e}")

with open(filename, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['timestamp', 'x', 'y'])
    for row in data:
        writer.writerow([f"{row[0]:.3f}", row[1], row[2]])

logging.info(f"Collection complete. Saved {len(data)} points to {filename}.")