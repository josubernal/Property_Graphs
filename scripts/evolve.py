from neo4j import GraphDatabase
from dotenv import load_dotenv
import os
import pandas as pd

load_dotenv()
URI = os.environ.get('NEO4J_URI')
USERNAME = os.environ.get('NEO4J_USERNAME')
PASSWORD = os.environ.get('NEO4J_PASSWORD')
DB_PATH= os.environ.get('NEO4J_DB_PATH')

driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))

def run_query(query, params={}):
    with driver.session() as session:
        result = session.run(query, params)
        return result.data()  # Returns list of dictionaries
    
import shutil
import os

source = os.getcwd().replace("\\", "/") + "/csv/affiliation.csv"
destination = DB_PATH + "/import"

shutil.copy(source, destination) 

file="csv/reviews.csv"
df=pd.read_csv(file, sep="|")
bad_rev=df[df["type"]=="bad"]["review"].to_list()
good_rev=df[df["type"]=="good"]["review"].to_list()


#I use modulos instead of rand() to make it reproducible (Seeds are not supported)

queries=["""
LOAD CSV FROM 'file:///affiliation.csv' AS row
FIELDTERMINATOR '|'
MERGE (n:affiliation {affiliationId: toInteger(row[0]), name: row[1], type: row[2]})
""",
"""
MATCH (a:author)
MATCH (af:affiliation)
WITH a, COLLECT(af) as affs, SIZE(COLLECT(af)) as total
WITH a, affs[ID(a) % total] as m
MERGE (a)-[:affiliated_to]->(m)
""",
"""
MATCH (n:journal|event)
SET n.reviewerNumber = CASE
    WHEN ID(n)%3 = 0 THEN 1 
    WHEN ID(n)%3 = 1 THEN 3
    ELSE 5
END
""",
"""
MATCH (j:journal)-[:published]-(p:paper)-[c:reviewed_by]-(a:author)
WITH j, p, collect(c) AS list
WHERE j.reviewerNumber = 1
DELETE  list[1],list[2]

UNION 

MATCH (j:journal)-[:published]-(p:paper)-[c:reviewed_by]-(a:author)
WHERE j.reviewerNumber = 5
MATCH (au:author)
WHERE NOT (au)-[:writes|reviewed_by]-(p:paper)
WITH p, collect(au)[0..2] as authors
UNWIND authors AS auth
MERGE (p)-[:reviewed_by]->(auth)
""",
"""
MATCH (j:journal)-[:published]-(p:paper)-[c:reviewed_by]-(a:author)
SET c.resolution="Rejected"

UNION 

MATCH (j:journal)-[:published]-(p:paper)-[c:reviewed_by]-(a:author)
WHERE j.reviewerNumber = 1
SET c.resolution="Accepted"

UNION

MATCH (j:journal)-[:published]-(p:paper)-[c:reviewed_by]-(a:author)
WHERE j.reviewerNumber = 3
WITH p, collect(c)[0..((ID(p)%2)+2)] as approved
UNWIND approved AS appr
SET appr.resolution="Accepted"

UNION

MATCH (j:journal)-[:published]-(p:paper)-[c:reviewed_by]-(a:author)
WHERE j.reviewerNumber = 5
WITH p, collect(c)[0..((ID(p)%3)+3)] as approved
UNWIND approved AS appr
SET appr.resolution="Accepted"
""",
f"""
MATCH (:paper)-[r:reviewed_by]-(:author)
WITH r, {bad_rev} AS bad_rev, SIZE({bad_rev}) AS bad_total, {good_rev} AS good_rev, SIZE({good_rev}) AS good_total
SET r.review = CASE
    WHEN r.resolution="Accepted" THEN good_rev[ID(r) % good_total]
    ELSE bad_rev[ID(r) % bad_total]
END
"""
]

for query in queries:
    run_query(query)