# D -- PR -- F -- PR -- I
import sys

from hrank import row_normalize, HRank_ASym, HRank_Sym, mean_time_of_decision
from question import pr_accepted, pr_rejected, review_time_less_than_mean_time_accepted, review_time_less_than_mean_time, review_time_more_than_mean_time, review_time_more_than_mean_time_accepted
from pymongo import MongoClient
import numpy as np


def Metapath9(collection, project_name):
    global db, project, dev_names, inte_names, U_dev_pr, U_pr_dev, U_pr_file, U_file_pr, U_pr_int, U_int_pr
    db = MongoClient("mongodb://localhost:27017/")[collection]


    project = project_name
    # Fetching pull requests of project giraph
    project_id = list(db["project"].find({"name" : project}, {"_id" : 1}))[0]["_id"]

    pull_request_system_id = list(db["pull_request_system"].find({"project_id":project_id}, {"_id": 1}))[0]["_id"]
    vcs_system_id = list(db["vcs_system"].find({"project_id":project_id}, {"_id": 1}))[0]["_id"]
    issue_system_id = list(db["issue_system"].find({"project_id":project_id}, {"_id":1}))[0]["_id"]


    # Fetching the developers of that project with pull request
    devs = list(db["pull_request"].find({"pull_request_system_id":pull_request_system_id}, {"creator_id":1, "_id":0}))
    devs = [x["creator_id"] for x in devs]
    devs = set(devs)              # removing duplicates

    devs2 = list(db["commit"].find({"vcs_system_id":vcs_system_id}, {"author_id":1, "committer_id":1}))
    devs2 = [x["author_id"] for x in devs2]+[x["committer_id"] for x in devs2]
    devs2 = set(devs2)
    devs = devs.union(devs2)

    devs2 = list(db["issue"].find({"issue_system_id":issue_system_id}, {"creator_id":1}))
    devs3 = []
    for x in devs2:
        if 'creator_id' in x:
            devs3.append(x['creator_id'])
    devs3 = set(devs3)
    devs = devs.union(devs3)
    devs = list(devs)
    dev_indexes = {k:v for v, k in enumerate(devs)}

    dev_names = []
    for x in devs:
        person = list(db["people"].find({"_id":x}, {"name":1, "_id":0}))[0]["name"]
        dev_names.append(person)



    # Fetching PRs
    prs = list(db["pull_request"].find({"pull_request_system_id":pull_request_system_id}, {"_id":1}))
    prs = [x["_id"] for x in prs]
    pr_indexes = {k:v for v, k in enumerate(prs)}



    # Fetching files
    files = list(db["file"].find({"vcs_system_id":vcs_system_id}, {"_id":1}))
    files = [x['_id'] for x in files]
    files = set(files)
    file_indexes = {k:v for v, k in enumerate(files)}



    # Fetching integrators
    integrators = set()
    merged_prs = list(db["pull_request_event"].find({"event_type":"merged"}, {"author_id":1, "pull_request_id":1}))
    for x in merged_prs:
        if x['pull_request_id'] in pr_indexes:
            integrators.add(x['author_id'])
    integrators = list(integrators)
    inte_indexes = {k:v for v, k in enumerate(integrators)}

    inte_names = []
    for x in integrators:
        person = list(db["people"].find({"_id": x}, {"name": 1, "_id": 0}))[0]["name"]
        inte_names.append(person)

    if(len(integrators)==0):
        sys.exit("No integrators found")


    # dev->PR and PR->dev
    M_dev_pr = np.zeros((len(devs), len(prs)))
    for i in range(len(devs)):
        pr_by_dev = list(db["pull_request"].find({"pull_request_system_id":pull_request_system_id,
                                                  "creator_id": devs[i]}, {"_id":1}))
        pr_by_dev = [x["_id"] for x in pr_by_dev]
        for x in pr_by_dev:
            M_dev_pr[i][pr_indexes[x]] = 1


    path_to_file = {}
    for x in files:
        path = list(db["file"].find({"_id":x}, {"path":1, "_id":1}))[0]['path']
        path_to_file[path] = file_indexes[x]


    # PR->file and file->PR
    M_pr_file = np.zeros((len(prs), len(files)))
    for i in range(len(prs)):
        file_by_pr = list(db["pull_request_file"].find({"pull_request_id": prs[i]}, {"path":1}))
        file_by_pr = [x["path"] for x in file_by_pr]
        for x in file_by_pr:
            if x in path_to_file:
                M_pr_file[i][path_to_file[x]] = 1



    # PR->int
    M_pr_int = np.zeros((len(prs), len(integrators)))
    for i in range(len(prs)):
        int_by_pr = list(db["pull_request_event"].find({"pull_request_id":prs[i], "event_type":"merged"},
                                                       {"pull_request_id":1, "author_id":1}))
        for x in int_by_pr:
            if x['author_id'] in inte_indexes:
                M_pr_int[i][inte_indexes[x['author_id']]] = 1



    U_dev_pr = row_normalize(M_dev_pr)
    U_pr_file = row_normalize(M_pr_file)
    U_pr_int = row_normalize(M_pr_int)

    M_pr_dev = np.transpose(M_dev_pr)
    M_file_pr = np.transpose(M_pr_file)
    M_int_pr = np.transpose(M_pr_int)

    U_pr_dev = row_normalize(M_pr_dev)
    U_file_pr = row_normalize(M_file_pr)
    U_int_pr = row_normalize(M_int_pr)


### Questions

### 1

def m9_q1(project):
    Metapath9("smartshark", project)
    metapath1 = np.matmul(U_dev_pr, np.matmul(U_pr_file, np.matmul(U_file_pr, U_pr_int)))
    metapath2 = np.matmul(U_int_pr, np.matmul(U_pr_file, np.matmul(U_file_pr, U_pr_dev)))

    metapath1 = row_normalize(metapath1)
    metapath2 = row_normalize(metapath2)

    rank, _ = HRank_ASym(dev_names, inte_names, metapath1, metapath2)

    return rank



### 2
def m9_q2(project):
    Metapath9("smartshark", project)
    M_pr_pr = pr_accepted(project)
    metapath1 = np.matmul(U_dev_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr,
    np.matmul(U_pr_file, np.matmul(U_file_pr, U_pr_int)))))
    metapath2 = np.matmul(U_int_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr,
    np.matmul(U_pr_file, np.matmul(U_file_pr, U_pr_dev)))))

    metapath1 = row_normalize(metapath1)
    metapath2 = row_normalize(metapath2)

    rank, _ = HRank_ASym(dev_names, inte_names, metapath1, metapath2)

    return rank


### 3
def m9_q3(project):
    Metapath9("smartshark", project)
    M_pr_pr = pr_rejected(project)

    metapath1 = np.matmul(U_dev_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr,
    np.matmul(U_pr_file, np.matmul(U_file_pr, U_pr_int)))))
    metapath2 = np.matmul(U_int_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr,
    np.matmul(U_pr_file, np.matmul(U_file_pr, U_pr_dev)))))

    metapath1 = row_normalize(metapath1)
    metapath2 = row_normalize(metapath2)

    rank, _ = HRank_ASym(dev_names, inte_names, metapath1, metapath2)

    return rank



### 7
def m9_q7(project):
    Metapath9("smartshark", project)
    decision = input("Enter decision (closed/merged/added_to_project) : ")
    M_pr_pr = review_time_less_than_mean_time(project, decision)

    metapath1 = np.matmul(U_dev_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr,
    np.matmul(U_pr_file, np.matmul(U_file_pr, U_pr_int)))))
    metapath2 = np.matmul(U_int_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr,
    np.matmul(U_pr_file, np.matmul(U_file_pr, U_pr_dev)))))

    metapath1 = row_normalize(metapath1)
    metapath2 = row_normalize(metapath2)

    rank,_ = HRank_ASym(dev_names, inte_names, metapath1, metapath2)

    return rank



### 8
def m9_q8(project):
    Metapath9("smartshark", project)
    decision = input("Enter decision (closed/merged/added_to_project) : ")
    M_pr_pr = review_time_more_than_mean_time(project, decision)

    metapath1 = np.matmul(U_dev_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr,
    np.matmul(U_pr_file, np.matmul(U_file_pr, U_pr_int)))))
    metapath2 = np.matmul(U_int_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr,
    np.matmul(U_pr_file, np.matmul(U_file_pr, U_pr_dev)))))

    metapath1 = row_normalize(metapath1)
    metapath2 = row_normalize(metapath2)

    rank,_ = HRank_ASym(dev_names, inte_names, metapath1, metapath2)

    return rank


### 9
def m9_q9(project):
    Metapath9("smartshark", project)
    M_pr_pr = review_time_less_than_mean_time_accepted(project)

    metapath1 = np.matmul(U_dev_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr,
    np.matmul(U_pr_file, np.matmul(U_file_pr, U_pr_int)))))
    metapath2 = np.matmul(U_int_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr,
    np.matmul(U_pr_file, np.matmul(U_file_pr, U_pr_dev)))))

    metapath1 = row_normalize(metapath1)
    metapath2 = row_normalize(metapath2)

    rank,_ = HRank_ASym(dev_names, inte_names, metapath1, metapath2)

    return rank


### 10
def m9_q10(project):
    Metapath9("smartshark", project)
    M_pr_pr = review_time_more_than_mean_time_accepted(project)

    metapath1 = np.matmul(U_dev_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr,
    np.matmul(U_pr_file, np.matmul(U_file_pr, U_pr_int)))))
    metapath2 = np.matmul(U_int_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr,
    np.matmul(U_pr_file, np.matmul(U_file_pr, U_pr_dev)))))

    metapath1 = row_normalize(metapath1)
    metapath2 = row_normalize(metapath2)

    rank,_ = HRank_ASym(dev_names, inte_names, metapath1, metapath2)

    return rank

project = input("Enter project name: ")
print("Question 1: ", end='')
print(m9_q1(project))

print("Question 2: ", end='')
print(m9_q2(project))

print("Question 3: ", end='')
print(m9_q3(project))

print("Question 7: ", end='')
print(m9_q7(project))

print("Question 8: ", end='')
print(m9_q8(project))

print("Question 9: ", end='')
print(m9_q9(project))

print("Question 10: ", end='')
print(m9_q10(project))




