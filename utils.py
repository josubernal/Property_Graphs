from neo4j import GraphDatabase
from dotenv import load_dotenv
import os

class Driver:
    def __init__(self):
        load_dotenv()
        uri = os.environ.get('NEO4J_URI')
        username = os.environ.get('NEO4J_USERNAME')
        password = os.environ.get('NEO4J_PASSWORD')
        self.driver = GraphDatabase.driver(uri, auth=(username, password))

    def run_query(self, query, params={}):
        with self.driver.session() as session:
            result = session.run(query, params)
            return result.data()  # Returns list of dictionaries