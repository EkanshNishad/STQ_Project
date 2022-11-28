import numpy as np
from pymongo import MongoClient
import sys

# 2
def pr_accepted(project):
    db = MongoClient("mongodb://localhost:27017/")["smartshark"]
    project_id = list(db["project"].find({"name": project}, {"_id": 1}))[0]["_id"]
    pull_request_system_id = list(db["pull_request_system"].find({"project_id": project_id}, {"_id": 1}))[0]["_id"]

    prs = list(db["pull_request"].find({"pull_request_system_id": pull_request_system_id}, {"_id": 1}))
    prs = [x["_id"] for x in prs]
    pr_indexes = {k: v for v, k in enumerate(prs)}

    M_pr_pr = np.zeros((len(prs), len(prs)))
    pr_accepted = list(db["pull_request_review"].find({"state": "APPROVED"}, {"pull_request_id": 1}))
    pr_accepted = [x['pull_request_id'] for x in pr_accepted]

    for x in pr_accepted:
        if x in pr_indexes:
            M_pr_pr[pr_indexes[x]][pr_indexes[x]] = 1
    return M_pr_pr

# 3
def pr_rejected(project):
    db = MongoClient("mongodb://localhost:27017/")["smartshark"]
    project_id = list(db["project"].find({"name": project}, {"_id": 1}))[0]["_id"]
    pull_request_system_id = list(db["pull_request_system"].find({"project_id": project_id}, {"_id": 1}))[0]["_id"]

    prs = list(db["pull_request"].find({"pull_request_system_id": pull_request_system_id}, {"_id": 1}))
    prs = [x["_id"] for x in prs]
    pr_indexes = {k: v for v, k in enumerate(prs)}

    M_pr_pr = np.zeros((len(prs), len(prs)))
    pr_rejected = list(db["pull_request_review"].find({"state": "DISMISSED"}, {"pull_request_id": 1}))
    pr_rejected = [x['pull_request_id'] for x in pr_rejected]
    for x in pr_rejected:
        if x in pr_indexes:
            M_pr_pr[pr_indexes[x]][pr_indexes[x]] = 1
    return M_pr_pr

# 7
def review_time_less_than_mean_time(project, decision):
    db = MongoClient("mongodb://localhost:27017/")["smartshark"]
    project_id = list(db["project"].find({"name": project}, {"_id": 1}))[0]["_id"]
    pull_request_system_id = list(db["pull_request_system"].find({"project_id": project_id}, {"_id": 1}))[0]["_id"]

    prs = list(db["pull_request"].find({"pull_request_system_id": pull_request_system_id}, {"_id": 1}))
    prs = [x["_id"] for x in prs]
    pr_indexes = {k: v for v, k in enumerate(prs)}

    decision_list = list(db["pull_request_event"].find({"event_type": decision}, {"pull_request_id": 1, "_id": 1}))
    pr_event_ids = dict()
    # pr_ids_dict=dict()
    for p in decision_list:
        if p['pull_request_id'] in pr_indexes:
            pr_event_ids[p['_id']] = p['pull_request_id']

    sum_time = 0
    for p in pr_event_ids:
        t2 = (list(db["pull_request_event"].find({"_id": p}, {"created_at": 1, "_id": 0}))[0]["created_at"])
        t1 = (list(db["pull_request"].find({"_id": pr_event_ids[p]}, {"created_at": 1, "_id": 0}))[0]["created_at"])
        td = t2 - t1
        sum_time += (td.total_seconds())
    if len(pr_event_ids)==0:
        return "Decision not found", "Decision not found"
    mean_time = sum_time / len(pr_event_ids)

    prs_with_review_more_mean_time = list()
    for p in pr_event_ids:
        t1 = (list(db["pull_request"].find({"_id": pr_event_ids[p]}, {"created_at": 1, "_id": 0}))[0]["created_at"])
        try:
            t2 = list(db["pull_request_review"].find({"pull_request_id": pr_event_ids[p]},
                                                     {"submitted_at": 1, "_id": 0}).sort("submitted_at", 1))
            reviews_time = t2[len(t2) - 1]["submitted_at"] - t1
            if (reviews_time).total_seconds() < mean_time:
                prs_with_review_more_mean_time.append(pr_event_ids[p])
        except:
            exception = True

    M_pr_pr = np.zeros((len(prs), len(prs)))
    for x in prs_with_review_more_mean_time:
        if x in pr_indexes:
            M_pr_pr[pr_indexes[x]][pr_indexes[x]] = 1

    return M_pr_pr


# 8
def review_time_more_than_mean_time(project,decision):
    db = MongoClient("mongodb://localhost:27017/")["smartshark"]
    project_id = list(db["project"].find({"name": project}, {"_id": 1}))[0]["_id"]
    pull_request_system_id = list(db["pull_request_system"].find({"project_id": project_id}, {"_id": 1}))[0]["_id"]

    prs = list(db["pull_request"].find({"pull_request_system_id": pull_request_system_id}, {"_id": 1}))
    prs = [x["_id"] for x in prs]

    M_pr_pr=review_time_less_than_mean_time(project,decision)
    if M_pr_pr[0] == "Decision not found":
        return M_pr_pr
    for i in range(len(prs)):
        M_pr_pr[i][i]=not M_pr_pr[i][i]
    return M_pr_pr


# 9
def review_time_less_than_mean_time_accepted(project):
    db = MongoClient("mongodb://localhost:27017/")["smartshark"]
    project_id = list(db["project"].find({"name": project}, {"_id": 1}))[0]["_id"]
    pull_request_system_id = list(db["pull_request_system"].find({"project_id": project_id}, {"_id": 1}))[0]["_id"]

    prs = list(db["pull_request"].find({"pull_request_system_id": pull_request_system_id}, {"_id": 1}))
    prs = [x["_id"] for x in prs]
    pr_indexes = {k: v for v, k in enumerate(prs)}

    approved_prs = set()
    for p in prs:
        approved = list(db["pull_request_review"].find({"pull_request_id": p, "state": "APPROVED"}))
        if len(approved) != 0:
            approved_prs.add(p)

    pr_event_ids = dict()
    # pr_ids_dict=dict()
    for p in prs:
        if p in approved_prs:
            try:
                e_id = (list(db["pull_request_event"].find({"pull_request_id": p}, {"_id": 1}))[0]["_id"])
                pr_event_ids[e_id] = p
                # pr_ids_dict[p]=e_id
            except:
                exception = True

    sum_time = 0
    for p in prs:
        t2 = list(
            db["pull_request_review"].find({"pull_request_id": p, "state": "APPROVED"}, {"submitted_at": 1, "_id": 0}))
        if len(t2) == 0:
            continue
        t2 = t2[0]["submitted_at"]
        t1 = (list(db["pull_request"].find({"_id": p}, {"created_at": 1, "_id": 0}))[0]["created_at"])
        td = t2 - t1
        sum_time += (td.total_seconds())

    if len(pr_event_ids)==0:
        return
    mean_time = sum_time / len(pr_event_ids)

    if len(pr_event_ids)==0:
        return "Decision not found", "Decision not found"

    prs_with_review_more_mean_time = list()
    for p in pr_event_ids:
        t1 = (list(db["pull_request"].find({"_id": pr_event_ids[p]}, {"created_at": 1, "_id": 0}))[0]["created_at"])
        try:
            t2 = list(db["pull_request_review"].find({"pull_request_id": pr_event_ids[p]},
                                                     {"submitted_at": 1, "_id": 0}).sort("submitted_at", 1))
            reviews_time = t2[len(t2) - 1]["submitted_at"] - t1
            if (reviews_time).total_seconds() < mean_time:
                prs_with_review_more_mean_time.append(pr_event_ids[p])
        except:
            exception = True

    M_pr_pr = np.zeros((len(prs), len(prs)))
    for x in prs_with_review_more_mean_time:
        if x in pr_indexes:
            M_pr_pr[pr_indexes[x]][pr_indexes[x]] = 1

    return M_pr_pr


def review_time_more_than_mean_time_accepted(project):
    db = MongoClient("mongodb://localhost:27017/")["smartshark"]
    project_id = list(db["project"].find({"name": project}, {"_id": 1}))[0]["_id"]
    pull_request_system_id = list(db["pull_request_system"].find({"project_id": project_id}, {"_id": 1}))[0]["_id"]

    prs = list(db["pull_request"].find({"pull_request_system_id": pull_request_system_id}, {"_id": 1}))
    prs = [x["_id"] for x in prs]
    pr_indexes = {k: v for v, k in enumerate(prs)}

    M_pr_pr=review_time_less_than_mean_time_accepted(project)
    if M_pr_pr[0] == "Decision not found":
        return M_pr_pr
    for i in range(len(prs)):
        M_pr_pr[i][i]=not M_pr_pr[i][i]
    return M_pr_pr