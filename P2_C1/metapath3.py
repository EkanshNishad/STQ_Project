# project - giraph
# D -- PR -- PRC -- D
from hrank import row_normalize, HRank_ASym, HRank_Sym, mean_time_of_decision
from question import pr_accepted, pr_rejected, review_time_less_than_mean_time_accepted, review_time_less_than_mean_time, review_time_more_than_mean_time, review_time_more_than_mean_time_accepted
from pymongo import MongoClient
import numpy as np


def Metapath3(collection, project_name):
    global db, project, dev_names, U_dev_pr, U_pr_dev, U_pr_prc, U_prc_pr, U_prc_dev, U_dev_prc
    db = MongoClient("mongodb://localhost:27017/")["smartshark"]


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
    pr_c = []
    for pr in prs:
        com = list(db["pull_request_comment"].find({"pull_request_id":pr}, {"_id":1}))
        com = [x["_id"] for x in com]
        pr_c= pr_c+com

    pr_c = list(set(pr_c))
    prc_indexes = {k:v for v, k in enumerate(pr_c)}



    # dev->PR
    M_dev_pr = np.zeros((len(devs), len(prs)))
    for i in range(len(devs)):
        pr_by_dev = list(db["pull_request"].find({"pull_request_system_id":pull_request_system_id,
                                                  "creator_id": devs[i]}, {"_id":1}))
        pr_by_dev = [x["_id"] for x in pr_by_dev]
        for x in pr_by_dev:
            M_dev_pr[i][pr_indexes[x]] = 1


    U_dev_pr = row_normalize(M_dev_pr)


    # PR->PRC
    M_pr_prc = np.zeros((len(prs), len(pr_c)))
    for i in range(len(prs)):
        comment_by_pr = list(db["pull_request_comment"].find({"pull_request_id": prs[i]}))
        comment_by_pr = [x["_id"] for x in comment_by_pr]
        for x in comment_by_pr:
            M_pr_prc[i][prc_indexes[x]] = 1

    U_pr_prc = row_normalize(M_pr_prc)


    # PRC->dev
    M_prc_dev = np.zeros((len(pr_c), len(devs)))
    for i in range(len(pr_c)):
        dev_of_prc = list(db["pull_request_comment"].find({"_id": pr_c[i]}))
        dev_of_prc = [x["author_id"] for x in dev_of_prc]

        for x in dev_of_prc:
            if x in dev_indexes:
                M_prc_dev[i][dev_indexes[x]] = 1

    U_prc_dev = row_normalize(M_prc_dev)

    M_dev_prc = np.transpose(M_prc_dev)
    M_prc_pr = np.transpose(M_pr_prc)
    M_pr_dev = np.transpose(M_dev_pr)

    U_dev_prc = row_normalize(M_dev_prc)
    U_prc_pr = row_normalize(M_prc_pr)
    U_pr_dev = row_normalize(M_pr_dev)


### Questions


### 1

def m3_q1(project):
    Metapath3("smartshark", project)
    metapath1 = np.matmul(U_dev_pr, np.matmul(U_pr_prc, U_prc_dev))
    metapath2 = np.matmul(U_dev_prc, np.matmul(U_prc_pr, U_pr_dev))

    rank,_ = HRank_ASym(dev_names, dev_names, metapath1, metapath2)

    return rank



### 2
def m3_q2(project):
    Metapath3("smartshark", project)
    M_pr_pr = pr_accepted(project)

    metapath1 = np.matmul(U_dev_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr, np.matmul(U_pr_prc, U_prc_dev))))
    metapath2 = np.matmul(U_dev_prc, np.matmul(U_prc_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr, U_pr_dev))))

    rank,_ = HRank_ASym(dev_names, dev_names, metapath1, metapath2)

    return rank



### 3
def m3_q3(project):
    Metapath3("smartshark", project)
    M_pr_pr = pr_rejected(project)

    metapath1 = np.matmul(U_dev_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr, np.matmul(U_pr_prc, U_prc_dev))))
    metapath2 = np.matmul(U_dev_prc, np.matmul(U_prc_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr, U_pr_dev))))

    rank,_ = HRank_ASym(dev_names, dev_names, metapath1, metapath2)

    return rank



### 7
def m3_q7(project):
    Metapath3("smartshark", project)
    decision = input("Enter decision (closed/merged/added_to_project) : ")
    M_pr_pr = review_time_less_than_mean_time(project, decision)

    metapath1 = np.matmul(U_dev_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr, np.matmul(U_pr_prc, U_prc_dev))))
    metapath2 = np.matmul(U_dev_prc, np.matmul(U_prc_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr, U_pr_dev))))

    rank,_ = HRank_ASym(dev_names, dev_names, metapath1, metapath2)

    return rank



### 8
def m3_q8(project):
    Metapath3("smartshark", project)
    decision = input("Enter decision (closed/merged/added_to_project) : ")
    M_pr_pr = review_time_more_than_mean_time(project, decision)

    metapath1 = np.matmul(U_dev_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr, np.matmul(U_pr_prc, U_prc_dev))))
    metapath2 = np.matmul(U_dev_prc, np.matmul(U_prc_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr, U_pr_dev))))

    rank,_ = HRank_ASym(dev_names, dev_names, metapath1, metapath2)

    return rank



### 9
def m3_q9(project):
    Metapath3("smartshark", project)
    M_pr_pr = review_time_less_than_mean_time_accepted(project)

    metapath1 = np.matmul(U_dev_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr, np.matmul(U_pr_prc, U_prc_dev))))
    metapath2 = np.matmul(U_dev_prc, np.matmul(U_prc_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr, U_pr_dev))))

    rank,_ = HRank_ASym(dev_names, dev_names, metapath1, metapath2)

    return rank



### 10
def m3_q10(project):
    Metapath3("smartshark", project)
    M_pr_pr = review_time_more_than_mean_time_accepted(project)

    metapath1 = np.matmul(U_dev_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr, np.matmul(U_pr_prc, U_prc_dev))))
    metapath2 = np.matmul(U_dev_prc, np.matmul(U_prc_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr, U_pr_dev))))

    rank,_ = HRank_ASym(dev_names, dev_names, metapath1, metapath2)

    return rank

project = input("Enter project name: ")
print("Question 1: ", end='')
print(m3_q1(project))

print("Question 2: ", end='')
print(m3_q2(project))

print("Question 3: ", end='')
print(m3_q3(project))

print("Question 7: ", end='')
print(m3_q7(project))

print("Question 8: ", end='')
print(m3_q8(project))

print("Question 9: ", end='')
print(m3_q9(project))

print("Question 10: ", end='')
print(m3_q10(project))



