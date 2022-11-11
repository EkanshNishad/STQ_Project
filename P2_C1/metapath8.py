# D -- PR -- F -- PR -- PRrev -- R
from hrank import row_normalize, HRank_ASym, HRank_Sym, mean_time_of_decision
from question import pr_accepted, pr_rejected, review_time_less_than_mean_time_accepted, review_time_less_than_mean_time, review_time_more_than_mean_time, review_time_more_than_mean_time_accepted
from pymongo import MongoClient
import numpy as np

def Metapath8(collection, project_name):
    global db, project, dev_names, rev_names, U_dev_pr, U_pr_dev, U_pr_file, U_file_pr, U_prr_pr, U_pr_prr
    global U_prr_rev, U_rev_prr
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



    # Fetching reviews and reviewers
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

    path_to_file = {}
    for x in files:
        path = list(db["file"].find({"_id":x}, {"path":1}))[0]['path']
        path_to_file[path] = file_indexes[x]



    # Creating matrices for the metapath
    M_dev_pr = np.zeros((len(devs), len(prs)))
    for i in range(len(devs)):
        pr_by_dev = list(db["pull_request"].find({"pull_request_system_id": pull_request_system_id,
                                                  "creator_id": devs[i]}, {"_id:1"}))
        for x in pr_by_dev:
            M_dev_pr[i][pr_indexes[x['_id']]] = 1

    U_dev_pr = row_normalize(M_dev_pr)


    M_pr_file = np.zeros((len(prs), len(files)))
    for i in range(len(prs)):
        file_by_pr = list(db["pull_request_file"].find({"pull_request_id":prs[i]}, {"path":1, "_id":0}))
        for file in file_by_pr:
            if file['path'] in path_to_file:
                conv = path_to_file[file['path']]
                M_pr_file[i][conv] = 1

    U_pr_file = row_normalize(M_pr_file)


    M_pr_prr = np.zeros((len(prs), len(prrevs)))
    M_prr_rev = np.zeros((len(prrevs), len(reviewers)))
    for i in range(len(prs)):
        prr_by_pr = list(db["pull_request_review"].find({"pull_request_id":prs[i]}, {'_id':1, 'creator_id':1}))
        for x in prr_by_pr:
            M_pr_prr[i][prr_indexes[x['_id']]] = 1
            M_prr_rev[prr_indexes[x['_id']]][rev_indexes[x['creator_id']]] = 1

    U_pr_prr = row_normalize(M_pr_prr)
    U_prr_rev = row_normalize(M_prr_rev)


    M_rev_prr = np.transpose(M_prr_rev)
    M_prr_pr = np.transpose(M_pr_prr)
    M_file_pr = np.transpose(M_pr_file)
    M_pr_dev = np.transpose(M_dev_pr)

    U_rev_prr = row_normalize(M_rev_prr)
    U_prr_pr = row_normalize(M_prr_pr)
    U_file_pr = row_normalize(M_file_pr)
    U_pr_dev = row_normalize(M_pr_dev)


### Questions

### 1

def m8_q1(project):
    Metapath8("smartshark", project)
    metapath1 = np.matmul(U_dev_pr, np.matmul(U_pr_file, np.matmul(U_file_pr,
    np.matmul(U_pr_prr, U_prr_rev))))
    metapath2 = np.matmul(U_rev_prr, np.matmul(U_prr_pr, np.matmul(U_pr_file,
    np.matmul(U_file_pr, U_pr_dev))))

    metapath1 = row_normalize(metapath1)
    metapath2 = row_normalize(metapath2)

    rank,_ = HRank_ASym(dev_names, rev_names, metapath1, metapath2)
    return rank



### 2
def m8_q2(project):
    Metapath8("smartshark", project)
    M_pr_pr = pr_accepted(project)

    metapath1 = np.matmul(U_dev_pr, np.matmul(U_pr_file, np.matmul(U_file_pr,
    np.matmul(M_pr_pr, np.matmul(M_pr_pr, np.matmul(U_pr_prr, U_prr_rev))))))
    metapath2 = np.matmul(U_rev_prr, np.matmul(U_prr_pr, np.matmul(M_pr_pr,
    np.matmul(M_pr_pr, np.matmul(U_pr_file, np.matmul(U_file_pr, U_pr_dev))))))

    metapath1 = row_normalize(metapath1)
    metapath2 = row_normalize(metapath2)

    rank,_ = HRank_ASym(dev_names, rev_names, metapath1, metapath2)

    return rank



### 3
def m8_q3(project):
    Metapath8("smartshark", project)
    M_pr_pr = pr_rejected(project)

    metapath1 = np.matmul(U_dev_pr, np.matmul(U_pr_file, np.matmul(U_file_pr,
    np.matmul(M_pr_pr, np.matmul(M_pr_pr, np.matmul(U_pr_prr, U_prr_rev))))))
    metapath2 = np.matmul(U_rev_prr, np.matmul(U_prr_pr, np.matmul(M_pr_pr,
    np.matmul(M_pr_pr, np.matmul(U_pr_file, np.matmul(U_file_pr, U_pr_dev))))))

    metapath1 = row_normalize(metapath1)
    metapath2 = row_normalize(metapath2)

    rank,_ = HRank_ASym(dev_names, rev_names, metapath1, metapath2)

    return rank



### 4
def m8_q4(project):
    Metapath8("smartshark", project)
    metapath1 = np.matmul(U_dev_pr, np.matmul(U_pr_file, np.matmul(U_file_pr,
    np.matmul(U_pr_prr, U_prr_rev))))
    metapath2 = np.matmul(U_rev_prr, np.matmul(U_prr_pr, np.matmul(U_pr_file,
    np.matmul(U_file_pr, U_pr_dev))))

    metapath1 = row_normalize(metapath1)
    metapath2 = row_normalize(metapath2)

    _, rank = HRank_ASym(dev_names, rev_names, metapath1, metapath2)

    return rank



### 5
def m8_q5(project):
    Metapath8("smartshark", project)
    M_pr_pr = pr_accepted(project)

    metapath1 = np.matmul(U_dev_pr, np.matmul(U_pr_file, np.matmul(U_file_pr,
    np.matmul(M_pr_pr, np.matmul(M_pr_pr, np.matmul(U_pr_prr, U_prr_rev))))))
    metapath2 = np.matmul(U_rev_prr, np.matmul(U_prr_pr, np.matmul(M_pr_pr,
    np.matmul(M_pr_pr, np.matmul(U_pr_file, np.matmul(U_file_pr, U_pr_dev))))))

    metapath1 = row_normalize(metapath1)
    metapath2 = row_normalize(metapath2)

    _, rank = HRank_ASym(dev_names, rev_names, metapath1, metapath2)

    return rank



### 6
def m8_q6(project):
    Metapath8("smartshark", project)
    M_pr_pr = pr_rejected(project)

    metapath1 = np.matmul(U_dev_pr, np.matmul(U_pr_file, np.matmul(U_file_pr,
    np.matmul(M_pr_pr, np.matmul(M_pr_pr, np.matmul(U_pr_prr, U_prr_rev))))))
    metapath2 = np.matmul(U_rev_prr, np.matmul(U_prr_pr, np.matmul(M_pr_pr,
    np.matmul(M_pr_pr, np.matmul(U_pr_file, np.matmul(U_file_pr, U_pr_dev))))))

    metapath1 = row_normalize(metapath1)
    metapath2 = row_normalize(metapath2)

    _, rank = HRank_ASym(dev_names, rev_names, metapath1, metapath2)

    return rank



### 7
def m8_q7(project):
    Metapath8("smartshark", project)
    decision = input("Enter decision (closed/merged/added_to_project) : ")
    M_pr_pr = review_time_less_than_mean_time(project, decision)

    metapath1 = np.matmul(U_dev_pr, np.matmul(U_pr_file, np.matmul(U_file_pr,
    np.matmul(M_pr_pr, np.matmul(M_pr_pr, np.matmul(U_pr_prr, U_prr_rev))))))
    metapath2 = np.matmul(U_rev_prr, np.matmul(U_prr_pr, np.matmul(M_pr_pr,
    np.matmul(M_pr_pr, np.matmul(U_pr_file, np.matmul(U_file_pr, U_pr_dev))))))

    metapath1 = row_normalize(metapath1)
    metapath2 = row_normalize(metapath2)

    rank,_ = HRank_ASym(dev_names, rev_names, metapath1, metapath2)

    return rank


### 8
def m8_q8(project):
    Metapath8("smartshark", project)
    decision = input("Enter decision (closed/merged/added_to_project) : ")
    M_pr_pr = review_time_more_than_mean_time(project, decision)

    metapath1 = np.matmul(U_dev_pr, np.matmul(U_pr_file, np.matmul(U_file_pr,
    np.matmul(M_pr_pr, np.matmul(M_pr_pr, np.matmul(U_pr_prr, U_prr_rev))))))
    metapath2 = np.matmul(U_rev_prr, np.matmul(U_prr_pr, np.matmul(M_pr_pr,
    np.matmul(M_pr_pr, np.matmul(U_pr_file, np.matmul(U_file_pr, U_pr_dev))))))

    metapath1 = row_normalize(metapath1)
    metapath2 = row_normalize(metapath2)

    rank,_ = HRank_ASym(dev_names, rev_names, metapath1, metapath2)
    return rank



### 9
def m8_q9(project):
    Metapath8("smartshark", project)
    M_pr_pr = review_time_less_than_mean_time_accepted(project)

    metapath1 = np.matmul(U_dev_pr, np.matmul(U_pr_file, np.matmul(U_file_pr,
    np.matmul(M_pr_pr, np.matmul(M_pr_pr, np.matmul(U_pr_prr, U_prr_rev))))))
    metapath2 = np.matmul(U_rev_prr, np.matmul(U_prr_pr, np.matmul(M_pr_pr,
    np.matmul(M_pr_pr, np.matmul(U_pr_file, np.matmul(U_file_pr, U_pr_dev))))))

    metapath1 = row_normalize(metapath1)
    metapath2 = row_normalize(metapath2)

    rank,_ = HRank_ASym(dev_names, rev_names, metapath1, metapath2)

    return rank


### 10
def m8_q10(project):
    Metapath8("smartshark", project)
    M_pr_pr = review_time_more_than_mean_time_accepted(project)

    metapath1 = np.matmul(U_dev_pr, np.matmul(U_pr_file, np.matmul(U_file_pr,
    np.matmul(M_pr_pr, np.matmul(M_pr_pr, np.matmul(U_pr_prr, U_prr_rev))))))
    metapath2 = np.matmul(U_rev_prr, np.matmul(U_prr_pr, np.matmul(M_pr_pr,
    np.matmul(M_pr_pr, np.matmul(U_pr_file, np.matmul(U_file_pr, U_pr_dev))))))

    metapath1 = row_normalize(metapath1)
    metapath2 = row_normalize(metapath2)

    rank,_ = HRank_ASym(dev_names, rev_names, metapath1, metapath2)

    return rank


project = input("Enter project name: ")
print("Question 1: ", end='')
print(m8_q1(project))

print("Question 2: ", end='')
print(m8_q2(project))

print("Question 3: ", end='')
print(m8_q3(project))

print("Question 4: ", end='')
print(m8_q4(project))

print("Question 5: ", end='')
print(m8_q5(project))

print("Question 6: ", end='')
print(m8_q6(project))

print("Question 7: ", end='')
print(m8_q7(project))

print("Question 8: ", end='')
print(m8_q8(project))

print("Question 9: ", end='')
print(m8_q9(project))

print("Question 10: ", end='')
print(m8_q10(project))





