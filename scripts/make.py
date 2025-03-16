# %%
import requests, json, os, csv, re, urllib.parse
import pandas as pd
from dotenv import load_dotenv
import random
from time import sleep
from pathlib import Path
from contextlib import ExitStack
from datetime import datetime
import uuid
import numpy as np

from keybert import KeyBERT
from sentence_transformers import SentenceTransformer, util

load_dotenv()

api_key = os.environ.get('API_KEY')
headers = {"x-api-key": api_key}

def is_id_valid(paper):
    return paper is not None and not isinstance(paper, str) and paper['paperId'] is not None


def get_corresponding_author_id(authors):
    return authors[0]['authorId']


def is_valid_paper(paper):
    return paper['authors'] and get_corresponding_author_id(paper['authors']) and paper['abstract'] is not None and paper['title'] is not None and paper['year'] is not None and paper['publicationTypes'] is not None and paper["publicationVenue"] is not None


def is_valid_conference(paper):
    return "Conference" in paper["publicationTypes"] and paper["venue"] is not None 


def is_valid_journal(paper):
    return "JournalArticle" in paper["publicationTypes"] and paper["journal"] is not None and "name" in paper["journal"] and "pages" in paper["journal"] and "volume" in paper["journal"]


kw_model = KeyBERT()
mandatory_keywords = ["data management", "indexing", "data modeling", "big data", "data processing", "data storage", "data querying"]
semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
keyword_embeddings = semantic_model.encode(mandatory_keywords)
def generate_keywords(abstract):
    generated_keywords = kw_model.extract_keywords(abstract, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=2)
    generated_keywords = set(kw[0] for kw in generated_keywords)
    abstract_embedding = semantic_model.encode(abstract)
    top_indices = util.cos_sim(abstract_embedding, keyword_embeddings).argsort(descending=True)[0][:random.randint(0, 2)]

    best_mandatory_keywords = set([mandatory_keywords[idx] for idx in top_indices])
    return generated_keywords.union(best_mandatory_keywords)


csv_with_types = {
        "paper": ["paperId:ID", "corpusId:string", "title:string", "abstract:string", "url:string", "publicationDate:string", "year:int"],
        "paper_paper": ["citingPaperId:START_ID", "citedPaperId:END_ID"],
        "author": ["authorId:ID", "authorName:string"],
        "author_paper": ["authorId:START_ID", "paperId:END_ID"],
        "paper_corresponding_author": ["paperId:START_ID", "correspondingAuthorId:END_ID"],
        "paper_confws": ["paperId:START_ID", "confwsEditionId:END_ID"],
        "paper_journal": ["paperId:START_ID", "journalId:END_ID"],
        "keywords": ["keyword:ID"],
        "paper_keywords": ["paperId:START_ID", "keyword:END_ID"],
        "confws": ["confwsId:ID", "name:string", "label:LABEL"],
        "confws_edition": ["confwsEditionId:ID", "year:int", "city:string", "label:LABEL"],
        "confws_edition_confws": ["confwsEditionId:START_ID", "confwsId:END_ID"],
        "journal": ["journalId:ID", "journalName:string", "journalPages:string", "journalVolume:string","year:int"],
        "paper_review": ["paperId:START_ID", "reviewerAuthorId:END_ID"]
        }


csv_files = {}
final_csv_headers = {}
for key, value in csv_with_types.items():
    csv_files[key] = []
    final_csv_headers[key] = []
    for column in value:
        csv_files[key].append(column.split(":")[0])
        if "_ID" in column:
            final_csv_headers[key].append(':' + column.split(":")[1])
        else:
            final_csv_headers[key].append(column)


#********************************************************************************************************************
RECORDS = 100  # Number of records to save per category 
BATCH_SIZE = 300
MAX_RECURSION = 10
SEED_VALUE = 42
QUERY = "semantic data modelling and property graphs"  # Query to filter the papers
FIELDS = "paperId,corpusId,title,abstract,authors,url,year,publicationDate,publicationTypes,journal,venue,publicationVenue,references.paperId"  # Fields to retrieve from the API
#********************************************************************************************************************

query_encoded = urllib.parse.quote(QUERY)
fields_encoded = urllib.parse.quote(FIELDS)
type_encoded = urllib.parse.quote("Conference,JournalArticle")

starting_papers_url="https://api.semanticscholar.org/graph/v1/paper/search?query="+query_encoded+"&publicationTypes="+type_encoded+"&fields=paperId&limit="+str(RECORDS)
response = requests.get(starting_papers_url, headers=headers).json()
starting_papers = response["data"]


def process_new_papers(processed_papers, to_be_processed_papers, processing_papers, new_papers):
    new_papers.discard(None)
    for paper in new_papers:
        if paper not in processed_papers and paper not in to_be_processed_papers and paper not in processing_papers:
            to_be_processed_papers.add(paper)


def choose_n_papers_to_process(to_be_processed_papers, n):
    return {to_be_processed_papers.pop() for _ in range(min(n, len(to_be_processed_papers)))}

csv_folder = Path('csv')
cities = pd.read_csv('csv/city.csv', delimiter="|")
np.random.seed(SEED_VALUE) 

processed_papers = set()
to_be_processed_papers = set()
starting_papers_ids = set([paper['paperId'] for paper in starting_papers])
process_new_papers(processed_papers, to_be_processed_papers, set(), starting_papers_ids)

set_authors = set()
set_keywords = set()
set_papers = set()
set_journals = set()
set_confws = set()
set_confws_edition = set()


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
                    if random.random() > 0.5:
                        paper["publicationType"]="Conference"
                    else:
                        paper["publicationType"]="Workshop"
                    paper["journalName"]=None
                    paper["journalVolume"]=None
                    paper["journalPages"]=None
                    if paper["publicationId"] not in set_confws:
                        set_confws.add(paper["publicationId"])
                        writers["confws"].writerow({
                            "confwsId": paper["publicationId"],
                            "name": paper["publicationVenue"]["name"],
                            "label": paper["publicationType"]
                            })

                    confwsEditionId = paper["publicationId"] + str(paper["year"])
                    if confwsEditionId not in set_confws_edition:
                        set_confws_edition.add(confwsEditionId)
                        writers['confws_edition'].writerow({
                            "confwsEditionId": confwsEditionId,
                            "year": paper["year"],
                            "city": cities['city_ascii'].sample(n=1, replace=True).iloc[0],
                            "label": paper["publicationType"] + "Edition"
                            })

                        writers['confws_edition_confws'].writerow({
                            "confwsEditionId": confwsEditionId,
                            "confwsId": paper["publicationId"]
                            })

                    writers["paper_confws"].writerow({
                        "paperId": paperId,
                        "confwsEditionId": confwsEditionId
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
                    if paper["publicationId"]+str(paper["year"]) not in set_journals:
                        set_journals.add(paper["publicationId"]+str(paper["year"]))
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
                    "abstract": paper.get("abstract").strip().replace("\n", " ").replace("|", " ").replace('"', "").replace("^", " "),
                    "url": paper.get("url"),
                    "year": paper.get("year"),
                    "publicationDate": paper.get("publicationDate")
                    })


                writers['paper_corresponding_author'].writerow({
                    "paperId": paperId,
                    "correspondingAuthorId": get_corresponding_author_id(paper_authors)
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


                paper_keywords = generate_keywords(paper.get('abstract'))

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


#########################
#### POST-PROCESSING ####
#########################

# Cleaning references
papers_file = "csv/paper.csv"
citations_file="csv/paper_paper.csv"
df=pd.read_csv(papers_file, delimiter='|')

df_citations=pd.read_csv(citations_file, delimiter='|')
df_citations_new = df_citations[df_citations["citedPaperId"].isin(df['paperId'].to_list())]
df_citations_new.to_csv("csv/paper_paper.csv", sep='|', index=False)

authors_file = "csv/author.csv"
author_paper_file ="csv/author_paper.csv"


def get_unjoined_rows(dfpapers, dfauthors, dfauthorspapers, sample_size=3):
    result = []
    
    for i, row in dfpapers.iterrows():
        id1_value = row['paperId']
        
        joined_ids = set(dfauthorspapers[dfauthorspapers['paperId'] == id1_value]['authorId'])
        
        unjoined_dfauthors = dfauthors[~dfauthors['authorId'].isin(joined_ids)]
        
        sampled_rows = unjoined_dfauthors.sample(n=min(sample_size, len(unjoined_dfauthors)), random_state=42)
        
        for _, sample_row in sampled_rows.iterrows():
            result.append({'paperId': id1_value, 'reviewerId': sample_row['authorId']})
    
    return pd.DataFrame(result)

df_authors=pd.read_csv(authors_file, delimiter='|')
df_authors_papers=pd.read_csv(author_paper_file, delimiter='|')
df_reviwers= get_unjoined_rows(df, df_authors, df_authors_papers) 
df_reviwers.to_csv("csv/paper_review.csv", sep='|', index=False)




for csv_name in csv_files.keys():
    with open(csv_folder / (csv_name + '.csv'), 'r') as fin:
        data = fin.read().splitlines(True)
    with open(csv_folder / (csv_name + '.csv'), 'w') as fout:
        fout.write("|".join(final_csv_headers[csv_name]) + '\n')
        fout.writelines(data[1:])
        

print("CSVs successfully created")
