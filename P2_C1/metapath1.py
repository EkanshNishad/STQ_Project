# D -- F -- PR -- F -- D
from hrank import row_normalize, HRank_ASym, HRank_Sym, mean_time_of_decision
from question import pr_accepted, pr_rejected, review_time_less_than_mean_time_accepted, review_time_less_than_mean_time, review_time_more_than_mean_time, review_time_more_than_mean_time_accepted
from pymongo import MongoClient
import numpy as np

def Metapath1(collection, project_name):
    global db, project, dev_names, U_dev_file, U_file_pr, U_file_dev, U_pr_file
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



    # Fetching files
    files = list(db["file"].find({"vcs_system_id":vcs_system_id}, {"_id":1}))
    files = [x['_id'] for x in files]
    files = set(files)
    file_indexes = {k:v for v, k in enumerate(files)}



    # Fetching PRs
    prs = list(db["pull_request"].find({"pull_request_system_id":pull_request_system_id}, {"_id":1}))
    prs = [x["_id"] for x in prs]
    pr_indexes = {k:v for v, k in enumerate(prs)}



    # Creating matrices for metapath
    M_dev_file = np.zeros((len(devs), len(files)))
    M_file_pr = np.zeros((len(files), len(prs)))

    commits = list(db["commit"].find({"vcs_system_id":vcs_system_id}, {"author_id":1, "_id":1}))
    for cm in commits:
        file_by_cm = list(db["file_action"].find({"commit_id":cm['_id']}, {"file_id":1}))
        file_by_cm = set([x['file_id'] for x in file_by_cm])
        for x in file_by_cm:
            M_dev_file[dev_indexes[cm['author_id']]][file_indexes[x]] = 1


    path_to_file = {}
    for x in files:
        path = list(db["file"].find({"_id":x}, {"path":1}))[0]['path']
        path_to_file[path] = file_indexes[x]


    for i in range(len(prs)):
        file_by_pr = list(db["pull_request_file"].find({"pull_request_id":prs[i]}, {"path":1}))
        file_by_pr = [x['path'] for x in file_by_pr]
        for pf in file_by_pr:
            if pf in path_to_file:
                M_file_pr[path_to_file[pf]][i] = 1

    U_dev_file = row_normalize(M_dev_file)
    U_file_pr = row_normalize(M_file_pr)

    M_file_dev = np.transpose(M_dev_file)
    M_pr_file = np.transpose(M_file_pr)

    U_file_dev = row_normalize(M_file_dev)
    U_pr_file = row_normalize(M_pr_file)



### Questions


### 1
def m1_q1(project):
    Metapath1("smartshark", project)
    metapath = np.matmul(U_dev_file, np.matmul(U_file_pr, np.matmul(U_pr_file,
    U_file_dev)))
    metapath = row_normalize(metapath)

    rank = HRank_Sym(dev_names, metapath)

    return rank



### 2
def m1_q2(project):
    Metapath1("smartshark", project)
    M_pr_pr = pr_accepted(project)

    metapath = np.matmul(U_dev_file, np.matmul(U_file_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr,
    np.matmul(U_pr_file, U_file_dev)))))
    metapath = row_normalize(metapath)

    rank = HRank_Sym(dev_names, metapath)

    return rank



### 3
def m1_q3(project):
    Metapath1("smartshark", project)
    M_pr_pr = pr_rejected(project)

    metapath = np.matmul(U_dev_file, np.matmul(U_file_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr,
    np.matmul(U_pr_file, U_file_dev)))))
    metapath = row_normalize(metapath)

    rank = HRank_Sym(dev_names, metapath)

    return rank



### 7
def m1_q7(project):
    Metapath1("smartshark", project)
    decision = input("Enter decision (closed/merged/added_to_project) : ")
    M_pr_pr = review_time_less_than_mean_time(project, decision)

    metapath = np.matmul(U_dev_file, np.matmul(U_file_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr,
    np.matmul(U_pr_file, U_file_dev)))))
    metapath = row_normalize(metapath)

    rank = HRank_Sym(dev_names, metapath)

    return rank



### 8
def m1_q8(project):
    Metapath1("smartshark", project)
    decision = input("Enter decision (closed/merged/added_to_project) : ")
    M_pr_pr = review_time_more_than_mean_time(project, decision)
    metapath = np.matmul(U_dev_file, np.matmul(U_file_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr,
    np.matmul(U_pr_file, U_file_dev)))))
    metapath = row_normalize(metapath)

    rank = HRank_Sym(dev_names, metapath)

    return rank



### 9
def m1_q9(project):
    Metapath1("smartshark", project)
    M_pr_pr = review_time_less_than_mean_time_accepted(project)
    metapath = np.matmul(U_dev_file, np.matmul(U_file_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr,
    np.matmul(U_pr_file, U_file_dev)))))
    metapath = row_normalize(metapath)

    rank = HRank_Sym(dev_names, metapath)

    return rank


### 10
def m1_q10(project):
    M_pr_pr = review_time_more_than_mean_time_accepted(project)
    metapath = np.matmul(U_dev_file, np.matmul(U_file_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr,
    np.matmul(U_pr_file, U_file_dev)))))
    metapath = row_normalize(metapath)

    rank = HRank_Sym(dev_names, metapath)

    return rank

project = input("Enter project name: ")
print("Question 1: ", end='')
print(m1_q1(project))

print("Question 2: ", end='')
print(m1_q2(project))

print("Question 3: ", end='')
print(m1_q3(project))

print("Question 7: ", end='')
print(m1_q7(project))

print("Question 8: ", end='')
print(m1_q8(project))

print("Question 9: ", end='')
print(m1_q9(project))

print("Question 10: ", end='')
print(m1_q10(project))

