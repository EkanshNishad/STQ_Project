# D -- PR -- PRreview -- D
from hrank import row_normalize, HRank_ASym, HRank_Sym, mean_time_of_decision
from question import pr_accepted, pr_rejected, review_time_less_than_mean_time_accepted, review_time_less_than_mean_time, review_time_more_than_mean_time, review_time_more_than_mean_time_accepted
from pymongo import MongoClient
import numpy as np


def Metapath4(collection, project_name):
    global db, project, dev_names, U_dev_pr, U_pr_dev, U_pr_prr, U_prr_pr, U_prr_dev, U_dev_prr
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
    pr_indexes = {k:v for v,k in enumerate(prs)}



    # Fetching reviews
    prrevs = set()
    for pr in prs:
        reviews = list(db["pull_request_review"].find({"pull_request_id":pr}, {"_id":1}))
        prr = set([x['_id'] for x in reviews])
        prrevs = prrevs.union(prr)

    prrevs = list(prrevs)
    prr_indexes = {k:v for v,k in enumerate(prrevs)}


    # Creating matrixes for metapath
    M_prr_dev = np.zeros((len(prrevs), len(devs)))
    for i in range(len(prrevs)):
        author = list(db["pull_request_review_comment"].find({"pull_request_review_id":prrevs[i]}, {"creator_id":1}))
        creator = []
        for x in author:
            if 'creator_id' in x:
                creator.append(x['creator_id'])
        for x in creator:
            if x in dev_indexes:
                M_prr_dev[i][dev_indexes[x]] = 1

    M_pr_prr = np.zeros((len(prs), len(prrevs)))
    for i in range(len(prs)):
        reviews = list(db["pull_request_review"].find({"pull_request_id":prs[i]}, {"_id":1}))
        reviews = [x['_id'] for x in reviews]
        for x in reviews:
            M_pr_prr[i][prr_indexes[x]] = 1

    M_dev_pr = np.zeros((len(devs), len(prs)))
    for i in range(len(prs)):
        authors = list(db["pull_request"].find({"_id":prs[i]},{"creator_id":1}))
        authors = [x['creator_id'] for x in authors]
        for x in authors:
            M_dev_pr[dev_indexes[x]][i] = 1

    U_dev_pr = row_normalize(M_dev_pr)
    U_pr_prr = row_normalize(M_pr_prr)
    U_prr_dev = row_normalize(M_prr_dev)

    M_dev_prr = np.transpose(M_prr_dev)
    M_prr_pr = np.transpose(M_pr_prr)
    M_pr_dev = np.transpose(M_dev_pr)

    U_dev_prr = row_normalize(M_dev_prr)
    U_prr_pr = row_normalize(M_prr_pr)
    U_pr_dev = row_normalize(M_pr_dev)


### Questions


### 1
def m4_q1(project):
    Metapath4("smartshark", project)
    metapath1 = np.matmul(U_dev_pr, np.matmul(U_pr_prr, U_prr_dev))
    metapath2 = np.matmul(U_dev_prr, np.matmul(U_prr_pr, U_pr_dev))

    rank, _ = HRank_ASym(dev_names, dev_names, metapath1, metapath2)

    return rank



### 2
def m4_q2(project):
    Metapath4("smartshark", project)
    M_pr_pr = pr_accepted(project)

    metapath1 = np.matmul(U_dev_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr, np.matmul(U_pr_prr, U_prr_dev))))
    metapath2 = np.matmul(U_dev_prr, np.matmul(U_prr_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr, U_pr_dev))))

    rank, _ = HRank_ASym(dev_names, dev_names, metapath1, metapath2)

    return rank



### 3
def m4_q3(project):
    Metapath4("smartshark", project)
    M_pr_pr = pr_rejected(project)

    metapath1 = np.matmul(U_dev_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr, np.matmul(U_pr_prr, U_prr_dev))))
    metapath2 = np.matmul(U_dev_prr, np.matmul(U_prr_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr, U_pr_dev))))

    rank, _ = HRank_ASym(dev_names, dev_names, metapath1, metapath2)

    return rank



### 7
def m4_q7(project):
    Metapath4("smartshark", project)
    decision = input("Enter decision (closed/merged/added_to_project) : ")
    M_pr_pr = review_time_less_than_mean_time(project, decision)

    metapath1 = np.matmul(U_dev_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr, np.matmul(U_pr_prr, U_prr_dev))))
    metapath2 = np.matmul(U_dev_prr, np.matmul(U_prr_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr, U_pr_dev))))

    rank,_ = HRank_ASym(dev_names, dev_names, metapath1, metapath2)

    return rank



### 8
def m4_q8(project):
    Metapath4("smartshark", project)
    decision = input("Enter decision (closed/merged/added_to_project) : ")
    M_pr_pr = review_time_more_than_mean_time(project, decision)

    metapath1 = np.matmul(U_dev_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr, np.matmul(U_pr_prr, U_prr_dev))))
    metapath2 = np.matmul(U_dev_prr, np.matmul(U_prr_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr, U_pr_dev))))

    rank,_ = HRank_ASym(dev_names, dev_names, metapath1, metapath2)

    return rank



### 9
def m4_q9(project):
    Metapath4("smartshark", project)
    M_pr_pr = review_time_less_than_mean_time_accepted(project)

    metapath1 = np.matmul(U_dev_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr, np.matmul(U_pr_prr, U_prr_dev))))
    metapath2 = np.matmul(U_dev_prr, np.matmul(U_prr_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr, U_pr_dev))))

    rank,_ = HRank_ASym(dev_names, dev_names, metapath1, metapath2)

    return rank



# 10
def m4_q10(project):
    Metapath4("smartshark", project)
    M_pr_pr = review_time_more_than_mean_time_accepted(project)

    metapath1 = np.matmul(U_dev_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr, np.matmul(U_pr_prr, U_prr_dev))))
    metapath2 = np.matmul(U_dev_prr, np.matmul(U_prr_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr, U_pr_dev))))

    rank,_ = HRank_ASym(dev_names, dev_names, metapath1, metapath2)

    return rank

project = input("Enter project name: ")
print("Question 1: ", end='')
print(m4_q1(project))

print("Question 2: ", end='')
print(m4_q2(project))

print("Question 3: ", end='')
print(m4_q3(project))

print("Question 7: ", end='')
print(m4_q7(project))

print("Question 8: ", end='')
print(m4_q8(project))

print("Question 9: ", end='')
print(m4_q9(project))

print("Question 10: ", end='')
print(m4_q10(project))



