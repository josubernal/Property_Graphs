# %%
import requests, json, os, csv, re, urllib.parse
import pandas as pd
from dotenv import load_dotenv
from time import sleep
from pathlib import Path
from contextlib import ExitStack
from datetime import datetime
import numpy as np


load_dotenv()

# %%
api_key = os.environ.get('API_KEY')
headers = {"x-api-key": api_key}

# %%
def is_id_valid(paper):
    return paper is not None and not isinstance(paper, str) and paper['paperId'] is not None

def is_valid_paper(paper):
    return paper['authors'] and paper['abstract'] is not None and paper['title'] is not None and paper['year'] is not None and paper['publicationTypes'] is not None and paper["publicationVenue"] is not None


def is_valid_conference(paper):
    return "Conference" in paper["publicationTypes"] and paper["venue"] is not None 


def is_valid_journal(paper):
    return "JournalArticle" in paper["publicationTypes"] and paper["journal"] is not None and "name" in paper["journal"] and "pages" in paper["journal"] and "volume" in paper["journal"]


def get_referencing_author_id(authors):
    return authors[0]['authorId']

# %% [markdown]
# # CSV Needed
# - paper (w abstract and relevant author) 
# - paper-paper (n-n)
# - author 
# - paper-author (n-n)
# - paper-reviewers (n-n)
# - keywords
# - paper-keywords (n-n)
# - conference (1-n)
# - journal (1-n)
# - year (1-n)

# %%
csv_files = {
    "paper": ["paperId","corpusId", "title", "referenceAuthorId", "abstract", "url", "publicationType", "publicationDate","publicationId","year"],
    "paper_paper": ["citingPaperId", "citedPaperId"],
    "author": ["authorId", "authorName"],
    "author_paper": ["authorId", "paperId"],
    "paper_relevant_author": ["paperId", "relevantAuthorId"],
    "paper_reviewer": ["paperId", "reviewAuthorId"],
    "paper_conference": ["paperId", "conferenceId"],
    "paper_journal": ["paperId", "journalId"],
    "keywords": ["keyword"],
    "paper_keywords": ["paperId", "keyword"],
    "conference": ["conferenceId", "conferenceName", "year", "cityId"],
    "journal": ["journalId", "journalName", "journalPages", "journalVolume","year"],
}


# %%
#********************************************************************************************************************
RECORDS = 100  # Number of records to save per category 
QUERY = "semantic data modelling and property graphs"  # Query to filter the papers
FIELDS = "paperId,corpusId,title,abstract,authors,url,year,s2FieldsOfStudy,publicationDate,publicationTypes,journal,venue,publicationVenue,references.paperId"  # Fields to retrieve from the API
#********************************************************************************************************************

query_encoded = urllib.parse.quote(QUERY)
fields_encoded = urllib.parse.quote(FIELDS)
type_encoded = urllib.parse.quote("Conference,JournalArticle")

starting_papers_url="https://api.semanticscholar.org/graph/v1/paper/search?query="+query_encoded+"&publicationTypes="+type_encoded+"&fields=paperId&limit="+str(RECORDS)
response = requests.get(starting_papers_url, headers=headers).json()
starting_papers = response["data"]


# %%
def process_new_papers(processed_papers, to_be_processed_papers, processing_papers, new_papers):
    new_papers.discard(None)
    for paper in new_papers:
        if paper not in processed_papers and paper not in to_be_processed_papers and paper not in processing_papers:
            to_be_processed_papers.add(paper)


def choose_n_papers_to_process(to_be_processed_papers, n):
    return {to_be_processed_papers.pop() for _ in range(min(n, len(to_be_processed_papers)))}

# %%
BATCH_SIZE = 200
MAX_RECURSION = 2
csv_folder = Path('csv')
cities = pd.read_csv('csv/city.csv', delimiter="|")
seed_value = 42
np.random.seed(seed_value) 

processed_papers = set()
to_be_processed_papers = set()
starting_papers_ids = set([paper['paperId'] for paper in starting_papers])
process_new_papers(processed_papers, to_be_processed_papers, set(), starting_papers_ids)

set_authors = set()
set_keywords = set()
set_papers = set()
set_joutnals = set()
set_conferences = set()

with ExitStack() as stack:  # Ensures all files are closed properly
    files = {name: stack.enter_context(open(csv_folder / (name + '.csv'), "w", newline='', encoding="utf-8")) for name in csv_files}
    writers = {name: csv.DictWriter(files[name], fieldnames=fieldnames, delimiter="|") for name, fieldnames in csv_files.items()}
    recursion_block = 0

    # Header has to be removed for tables and changed for relationships! Ask Alfio
    for writer in writers.values():
        writer.writeheader()

    while to_be_processed_papers:
        recursion_block+=1
        if recursion_block > MAX_RECURSION:
            break

        processing_papers_id = choose_n_papers_to_process(to_be_processed_papers, BATCH_SIZE)

        processing_papers_data = requests.post(
            'https://api.semanticscholar.org/graph/v1/paper/batch',
            params={'fields': FIELDS},
            json={"ids": list(processing_papers_id)},
            headers=headers
        ).json()
        
        for paper in processing_papers_data:
            try:  
                if not is_id_valid(paper):
                    continue   
                processed_papers.add(paper['paperId'])
                if not is_valid_paper(paper):
                    continue
                paperId = paper.get("paperId")
                paper["publicationId"]=paper["publicationVenue"]["id"]
                if is_valid_conference(paper):
                    paper["publicationType"]="Conference"
                    paper["journalName"]=None
                    paper["journalVolume"]=None
                    paper["journalPages"]=None
                    conferenceId = paper["publicationId"] + str(paper["year"])
                    if paper["publicationId"]+str(paper["year"]) not in set_conferences:
                        set_conferences.add(paper["publicationId"]+str(paper["year"]))
                        writers['conference'].writerow({
                        "conferenceId": conferenceId,
                        "conferenceName": paper["publicationVenue"]["name"],
                        "year": paper["year"],
                        "cityId": cities['id'].sample(n=1, replace=True).iloc[0]
                        })

                    writers["paper_conference"].writerow({
                        "paperId": paperId,
                        "conferenceId": conferenceId
                    })
                    
                elif is_valid_journal(paper):
                    paper["publicationType"]="JournalArticle"
                    paper["venue"]=None
                    paper["journalName"]=paper["journal"]["name"]
                    paper["journalVolume"]=paper["journal"]["volume"]
                    if paper["journal"]["pages"] is not None:
                        paper["journalPages"]=re.sub(r'\s+', '', paper["journal"]["pages"])
                    else:
                        paper["journalPages"]=None
                    if paper["publicationId"]+str(paper["year"]) not in set_joutnals:
                        set_joutnals.add(paper["publicationId"]+str(paper["year"]))
                        writers['journal'].writerow({
                        "journalId": paper["publicationId"] + str(paper["year"]),
                        "journalName": paper["journalName"],
                        "journalVolume": paper["journalVolume"],
                        "journalPages": paper["journalPages"], 
                        "year": paper["year"],
                        })

                    writers["paper_journal"].writerow({
                        "paperId": paperId,
                        "journalId": paper["publicationId"] + str(paper["year"])
                    })

                else:
                    continue   
                paper_authors = paper["authors"]
                writers['paper'].writerow({
                    "paperId": paperId,
                    "corpusId": paper.get("corpusId"),
                    "title":  paper.get("title").strip().replace("\n", " ").replace("|", " ").replace('"', "").replace("^", " "),
                    "referenceAuthorId": get_referencing_author_id(paper_authors),
                    "abstract": paper.get("abstract").strip().replace("\n", " ").replace("|", " ").replace('"', "").replace("^", " "),
                    "url": paper.get("url"),
                    "year": paper.get("year"),
                    "publicationType": paper.get("publicationType"),
                    "publicationDate": paper.get("publicationDate"),
                    "publicationId": paper.get("publicationId")
                })


                writers['paper_relevant_author'].writerow({
                    "paperId": paperId,
                    "relevantAuthorId": get_referencing_author_id(paper_authors)
                })

                new_papers = set([paper['paperId'] for paper in paper['references']])
                process_new_papers(processed_papers, to_be_processed_papers, processing_papers_id, new_papers)

                for new_paper in new_papers:
                    writers['paper_paper'].writerow({
                        "citingPaperId": paperId,
                        "citedPaperId": new_paper,
                    })

                for author in paper_authors:
                    authorId = author.get("authorId")
                    authorName = author.get("name")
                    if authorId and authorName:
                        writers['author_paper'].writerow({
                            "authorId": authorId,
                            "paperId": paperId
                        })

                    if authorId and authorName and authorId not in set_authors:
                        writers['author'].writerow({
                                "authorId": authorId,
                                "authorName": authorName
                        })

                        set_authors.add(authorId)
            
                paper_keywords = paper.get("s2FieldsOfStudy", [])
                paper_keywords = set(map(lambda x: x['category'], paper_keywords))

                for keyword in paper_keywords:
                    writers["paper_keywords"].writerow({
                            "paperId": paperId,
                            "keyword": keyword
                        })
                    
                    if keyword not in set_keywords:
                        writers["keywords"].writerow({
                            "keyword": keyword
                        })

                        set_keywords.add(keyword)
                
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")


# %%
papers_file = "csv/paper.csv"
conferences_file = "csv/conference.csv"
journals_file = "csv/journal.csv"

df=pd.read_csv(papers_file, delimiter='|')
df_conferences=pd.read_csv(conferences_file, delimiter='|')
df_journals=pd.read_csv(journals_file, delimiter='|')


# %%
conferences=list(set(df_conferences["conferenceId"].to_list()))

split_index = len(conferences) // 2

work_shops = conferences[split_index:]
conferences = conferences[:split_index]

df_conf = df_conferences[df_conferences['conferenceId'].isin(conferences)]
df_workshops= df_conferences[df_conferences['conferenceId'].isin(work_shops)]
df_workshops.rename(columns={'conferenceId': 'workshopId','conferenceName':'workshopName'})
df.loc[df['publicationId'].isin(work_shops), 'publicationType'] = 'WorkShop'

# Save each filtered DataFrame to a separate CSV file
df_conf.to_csv("csv/conference.csv", sep="|",index=False)
df_workshops.to_csv("csv/workshop.csv", sep="|", index=False)
df.to_csv("csv/paper.csv", sep="|", index=False)

# %%
#Cleaning references :)
citations_file="csv/paper_paper.csv"

df_citations=pd.read_csv(citations_file, delimiter='|')
df_citations_new = df_citations[df_citations["citedPaperId"].isin(df['paperId'].to_list())]
df_citations_new.to_csv("csv/paper_paper.csv", sep='|', index=False)


# %%
url = "https://api.semanticscholar.org/graph/v1/author/batch"
BATCH_SIZE = 500
author_file="csv/author.csv"  
query_params = {
    "fields": "name,url,paperCount,hIndex"
}

df=pd.read_csv(author_file,  delimiter='|')
ids=df["authorId"].values.tolist()
df_copy = df.copy()


with open(author_file, "w", newline='', encoding="utf-8") as outfile  :   
    csv_writer_1 = csv.DictWriter(outfile, fieldnames=["authorId", "url", "authorName", "paperCount", "hIndex"], delimiter="|")
    # Write the headers to the CSV files
    csv_writer_1.writeheader()
    for i in range(0, len(ids), BATCH_SIZE):
        batch = ids[i:i + BATCH_SIZE]
        data = {
        "ids": batch
        }
        response = requests.post(url, params=query_params, json=data, headers=headers).json()
        # Save the results to json file
        count=0
        for paper in response:
            count+=1
            try: 
                if paper is not None: 
                    paper_row = {
                                "authorId": paper.get("authorId"),
                                "url": paper.get("url"),
                                "authorName": paper.get("name"),
                                "paperCount": paper.get("paperCount"),
                                "hIndex": paper.get("hIndex")
                                }
                    # Write the row to CSV 1
                    csv_writer_1.writerow(paper_row)
                else:
                    paper_row = {
                                "authorId":df_copy.loc[df_copy['authorId'] == batch[count-1], "authorId"].iloc(0),
                                "url": None,
                                "authorName": df_copy.loc[df_copy['authorId'] == batch[count-1], "authorName"].iloc(0),
                                "paperCount": None,
                                "hIndex": None
                                }
                    # Write the row to CSV 1
                    csv_writer_1.writerow(paper_row)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")

print("CSVs successfully created")
