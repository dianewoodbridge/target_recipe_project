import numpy as np 

def precision_at(k, search_results, relevant_docs):
    actual_positives = len(relevant_docs)
    true_positives = 0
    for search_result in search_results[0:k]:
        if search_result in relevant_docs:
            true_positives += 1
    return true_positives / k 

def recall_at(k, search_results, relevant_docs):
    actual_positives = len(relevant_docs)
    true_positives = 0
    for search_result in search_results[0:k]:
        if search_result in relevant_docs:
            true_positives += 1
    return true_positives / actual_positives

def precision_till(k, ranked_list, relevant_docs):
    precision_list = []
    for i in range(1, k+1, 1):
        precision_list.append(precision_at(i, ranked_list, relevant_docs))
    return precision_list 

def recall_till(k, ranked_list, relevant_docs):
    recall_list = []
    for i in range(1, k+1, 1):
        recall_list.append(recall_at(i, ranked_list, relevant_docs))
    return recall_list

def avg_precision_till(k, ranker_lists, relevant_docs_list):
    precision_sum = np.zeros(k)
    for ranker_list, relevant_docs in zip(ranker_lists, relevant_docs_list):
        precision_sum += np.array(precision_till(k, ranker_list, relevant_docs))
    return precision_sum / len(relevant_docs_list)

def avg_recall_till(k, ranker_lists, relevant_docs_list):
    recall_sum = np.zeros(k)
    for ranker_list, relevant_docs in zip(ranker_lists, relevant_docs_list):
        recall_sum += np.array(recall_till(k, ranker_list, relevant_docs))
    return recall_sum / len(relevant_docs_list)

def average_precision(ranker_list, relevant_docs):
    states = [match in relevant_docs for match in ranker_list]
    k_values = np.where(states)[0] + 1
    if len(k_values) == 0:
        return 0
    precision_sum = 0
    for k in k_values:
        precision_sum += precision_at(k, ranker_list, relevant_docs)
    return precision_sum / len(k_values)

def mean_average_precision(ranker_lists, relevant_docs_list, return_ap = False):
    avg_precision_sum = 0
    ap_list = []
    for ranker_list, relevant_docs in zip(ranker_lists, relevant_docs_list):
        ap = average_precision(ranker_list, relevant_docs)
        ap_list.append(ap)
        avg_precision_sum += ap
    mean_avg_precision = avg_precision_sum / len(relevant_docs_list)
    if return_ap == True:
        return mean_avg_precision, ap_list
    return mean_avg_precision