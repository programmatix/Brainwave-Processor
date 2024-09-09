import os
from influxdb import InfluxDBClient
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def query_influxdb(sql_query):
    # Retrieve credentials from environment variables
    host = os.getenv('INFLUXDB_HOST')
    port = os.getenv('INFLUXDB_PORT')
    username = os.getenv('INFLUXDB_USERNAME')
    password = os.getenv('INFLUXDB_PASSWORD')
    database = os.getenv('INFLUXDB_DATABASE')

    # Connect to InfluxDB
    client = InfluxDBClient(host=host, port=port, username=username, password=password, database=database)

    # Run the SQL query
    result = client.query(sql_query)

    # Return the results
    return result
