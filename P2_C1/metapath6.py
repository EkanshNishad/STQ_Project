# D -- PR -- PRreview -- R

from hrank import row_normalize, HRank_ASym, HRank_Sym, mean_time_of_decision
from question import pr_accepted, pr_rejected, review_time_less_than_mean_time_accepted, review_time_less_than_mean_time, review_time_more_than_mean_time, review_time_more_than_mean_time_accepted
from pymongo import MongoClient
import numpy as np


def Metapath6(collection, project_name):
    global db, project, dev_names, rev_names, U_dev_pr, U_pr_dev, U_pr_prr, U_prr_pr, U_prr_rev, U_rev_prr
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



    # Fetching PR reviews and reviewers
    prrevs = set()
    reviewers = set()
    for pr in prs:
        reviews = list(db["pull_request_review"].find({"pull_request_id":pr}, {"_id":1, "creator_id":1}))
        prr = set([x['_id'] for x in reviews])
        rev = set()
        for x in reviews:
            if 'creator_id' in x:
                rev.add(x['creator_id'])
        prrevs = prrevs.union(prr)
        reviewers = reviewers.union(rev)

    prrevs = list(prrevs)
    reviewers = list(reviewers)
    prr_indexes = {k:v for v,k in enumerate(prrevs)}
    rev_indexes = {k:v for v,k in enumerate(reviewers)}

    rev_names = []
    for x in reviewers:
        person = list(db["people"].find({"_id":x}, {"name":1, "_id":0}))[0]["name"]
        rev_names.append(person)



    # Creating matrix for metapath
    M_dev_pr = np.zeros((len(devs), len(prs)))
    M_pr_prr = np.zeros((len(prs), len(prrevs)))
    M_prr_rev = np.zeros((len(prrevs), len(reviewers)))


    # dev->PR
    for i in range(len(devs)):
        pr_by_dev = list(db["pull_request"].find({"pull_request_system_id":pull_request_system_id,
                                                  "creator_id": devs[i]}, {"_id:1"}))
        pr_by_dev = [x["_id"] for x in pr_by_dev]
        for x in pr_by_dev:
            M_dev_pr[i][pr_indexes[x]] = 1



    #PR->PRreview->Reviewers
    for i in range(len(prs)):
        rev_pr = list(db["pull_request_review"].find({"pull_request_id":prs[i]}, {"creator_id":1, "_id":1}))
        for x in rev_pr:
            M_pr_prr[i][prr_indexes[x['_id']]] = 1
        for x in rev_pr:
            M_prr_rev[prr_indexes[x['_id']]][rev_indexes[x['creator_id']]] = 1


    U_dev_pr = row_normalize(M_dev_pr)
    U_pr_prr = row_normalize(M_pr_prr)
    U_prr_rev = row_normalize(M_prr_rev)


    M_pr_dev = np.transpose(M_dev_pr)
    M_prr_pr = np.transpose(M_pr_prr)
    M_rev_prr = np.transpose(M_prr_rev)

    U_pr_dev = row_normalize(M_pr_dev)
    U_prr_pr = row_normalize(M_prr_pr)
    U_rev_prr = row_normalize(M_rev_prr)


### Questions


### 1
def m6_q1(project):
    Metapath6("smartshark", project)
    metapath1 = np.matmul(U_dev_pr, np.matmul(U_pr_prr, U_prr_rev))
    metapath2 = np.matmul(U_rev_prr, np.matmul(U_prr_pr, U_pr_dev))

    rank,_ = HRank_ASym(dev_names, rev_names, metapath1, metapath2)

    return rank



### 2
def m6_q2(project):
    Metapath6("smartshark", project)
    M_pr_pr = pr_accepted(project)

    metapath1 = np.matmul(U_dev_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr, np.matmul(U_pr_prr, U_prr_rev))))
    metapath2 = np.matmul(U_rev_prr, np.matmul(U_prr_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr, U_pr_dev))))

    rank,_ = HRank_ASym(dev_names, rev_names, metapath1, metapath2)

    return rank



### 3
def m6_q3(project):
    Metapath6("smartshark", project)
    M_pr_pr = pr_rejected(project)

    metapath1 = np.matmul(U_dev_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr, np.matmul(U_pr_prr, U_prr_rev))))
    metapath2 = np.matmul(U_rev_prr, np.matmul(U_prr_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr, U_pr_dev))))

    rank,_ = HRank_ASym(dev_names, rev_names, metapath1, metapath2)
    return rank



### 4
def m6_q4(project):
    Metapath6("smartshark", project)
    metapath1 = np.matmul(U_dev_pr, np.matmul(U_pr_prr, U_prr_rev))
    metapath2 = np.matmul(U_rev_prr, np.matmul(U_prr_pr, U_pr_dev))

    _, rank = HRank_ASym(dev_names, rev_names, metapath1, metapath2)

    return rank



### 5
def m6_q5(project):
    Metapath6("smartshark", project)
    M_pr_pr = pr_accepted(project)

    metapath1 = np.matmul(U_dev_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr, np.matmul(U_pr_prr, U_prr_rev))))
    metapath2 = np.matmul(U_rev_prr, np.matmul(U_prr_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr, U_pr_dev))))

    _, rank = HRank_ASym(dev_names, rev_names, metapath1, metapath2)

    return rank



### 6
def m6_q6(project):
    Metapath6("smartshark", project)
    M_pr_pr = pr_rejected(project)

    metapath1 = np.matmul(U_dev_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr, np.matmul(U_pr_prr, U_prr_rev))))
    metapath2 = np.matmul(U_rev_prr, np.matmul(U_prr_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr, U_pr_dev))))

    _, rank = HRank_ASym(dev_names, rev_names, metapath1, metapath2)

    return rank



### 7
def m6_q7(project):
    Metapath6("smartshark", project)
    decision = input("Enter decision (closed/merged/added_to_project) : ")
    M_pr_pr = review_time_less_than_mean_time(project, decision)

    metapath1 = np.matmul(U_dev_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr, np.matmul(U_pr_prr, U_prr_rev))))
    metapath2 = np.matmul(U_rev_prr, np.matmul(U_prr_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr, U_pr_dev))))

    rank,_ = HRank_ASym(dev_names, rev_names, metapath1, metapath2)

    return rank



### 8
def m6_q8(project):
    Metapath6("smartshark", project)
    decision = input("Enter decision (closed/merged/added_to_project) : ")
    M_pr_pr = review_time_more_than_mean_time(project, decision)

    metapath1 = np.matmul(U_dev_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr, np.matmul(U_pr_prr, U_prr_rev))))
    metapath2 = np.matmul(U_rev_prr, np.matmul(U_prr_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr, U_pr_dev))))

    rank,_ = HRank_ASym(dev_names, rev_names, metapath1, metapath2)

    return rank



### 9
def m6_q9(project):
    Metapath6("smartshark", project)
    M_pr_pr = review_time_less_than_mean_time_accepted(project)

    metapath1 = np.matmul(U_dev_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr, np.matmul(U_pr_prr, U_prr_rev))))
    metapath2 = np.matmul(U_rev_prr, np.matmul(U_prr_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr, U_pr_dev))))

    rank,_ = HRank_ASym(dev_names, rev_names, metapath1, metapath2)

    return rank



### 10
def m6_q10(project):
    Metapath6("smartshark", project)
    M_pr_pr = review_time_more_than_mean_time_accepted(project)
    metapath1 = np.matmul(U_dev_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr, np.matmul(U_pr_prr, U_prr_rev))))
    metapath2 = np.matmul(U_rev_prr, np.matmul(U_prr_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr, U_pr_dev))))

    rank,_ = HRank_ASym(dev_names, rev_names, metapath1, metapath2)

    return rank

project = input("Enter project name: ")
print("Question 1: ", end='')
print(m6_q1(project))

print("Question 2: ", end='')
print(m6_q2(project))

print("Question 3: ", end='')
print(m6_q3(project))

print("Question 4: ", end='')
print(m6_q4(project))

print("Question 5: ", end='')
print(m6_q5(project))

print("Question 6: ", end='')
print(m6_q6(project))

print("Question 7: ", end='')
print(m6_q7(project))

print("Question 8: ", end='')
print(m6_q8(project))

print("Question 9: ", end='')
print(m6_q9(project))

print("Question 10: ", end='')
print(m6_q10(project))


