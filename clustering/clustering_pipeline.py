import pickle

import pandas as pd

from clustering.openai_embedding import get_embedding
from clustering.clusterings import xmeans_clustering,hdbscan_clustering




cluster_algorithm = hdbscan_clustering
def cluster_entities(search_results_dict, query):
    publications = search_results_dict["publications"]
    resources = search_results_dict["resources"]
    projects = search_results_dict["projects"]
    others = search_results_dict["others"]
    dataframe = pd.DataFrame(columns=["text", "_original"])

    df_ready_publications = {"text": [], "_original": []}
    for i, publication in enumerate(publications):
        this_text = publication.name + "\n" + publication.description + "\nKeywords: " + ', '.join(publication.keywords)
        if this_text:
            df_ready_publications["text"].append(this_text)
            df_ready_publications["_original"].append(publication)
    df_ready_publications = pd.DataFrame(df_ready_publications)

    df_ready_resources = {"text": [], "_original": []}
    for i, resource in enumerate(resources):
        this_text = resource.name + "\n" + resource.description + "\nKeywords: " + ', '.join(resource.keywords)
        if this_text:
            df_ready_resources["text"].append(this_text)
            df_ready_resources["_original"].append(resource)
    df_ready_resources = pd.DataFrame(df_ready_resources)

    df_ready_projects = {"text": [], "_original": []}
    for i, project in enumerate(projects):
        this_text = project.name + "\n" + project.description
        if this_text:
            df_ready_projects["text"].append(this_text)
            df_ready_projects["_original"].append(project)
    df_ready_projects = pd.DataFrame(df_ready_projects)

    df_ready_others = {"text": [], "_original": []}
    for i, other in enumerate(others):
        this_text = other.name + "\n" + other.description + "\nKeywords: " + ', '.join(other.keywords)
        if this_text:
            df_ready_others["text"].append(this_text)
            df_ready_others["_original"].append(other)
    df_ready_others = pd.DataFrame(df_ready_others)

    dataframe = pd.concat([df_ready_publications, df_ready_resources, df_ready_projects, df_ready_others])

    embeddings = get_embedding(list(dataframe["text"]))

    cluster_labels, hdbscan_obj = cluster_algorithm(embeddings)

    cluster_dict = {}
    for i, cluster_label in enumerate(cluster_labels):
        if cluster_label == -1:
            continue
        if cluster_label not in cluster_dict:
            cluster_dict[cluster_label] = []
        cluster_dict[cluster_label].append(dataframe["_original"].iloc[i])

    query_key = query[:10]
    pickle_save_path = f"clustering/cluster_results/{cluster_algorithm.__name__}_{query_key}.pickle"

    with open(pickle_save_path, "wb") as f:
        pickle.dump((cluster_dict,hdbscan_obj), f)
