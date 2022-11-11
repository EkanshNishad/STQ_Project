# D -- F -- PR -- PRreview -- R
from hrank import row_normalize, HRank_ASym, HRank_Sym, mean_time_of_decision
from question import pr_accepted, pr_rejected, review_time_less_than_mean_time_accepted, review_time_less_than_mean_time, review_time_more_than_mean_time, review_time_more_than_mean_time_accepted
from pymongo import MongoClient
import numpy as np

db = MongoClient("mongodb://localhost:27017/")["smartshark"]

project = input("Enter the project name: ")
# Fetching pull requests of project giraph
project_id = list(db["project"].find({"name" : project}, {"_id" : 1}))[0]["_id"]
print("Project ID: "+str(project_id))
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


# Creating matrices for metapath
M_dev_file = np.zeros((len(devs), len(files)))
M_file_pr = np.zeros((len(files), len(prs)))
M_pr_prr = np.zeros((len(prs), len(prrevs)))
M_prr_rev = np.zeros((len(prrevs), len(reviewers)))


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
        


for i in range(len(prs)):
    rev_pr = list(db["pull_request_review"].find({"pull_request_id":prs[i]}, {"creator_id":1, "_id":1}))
    for x in rev_pr:
        M_pr_prr[i][prr_indexes[x['_id']]] = 1
    for x in rev_pr:
        M_prr_rev[prr_indexes[x['_id']]][rev_indexes[x['creator_id']]] = 1


U_dev_file = row_normalize(M_dev_file)
U_file_pr = row_normalize(M_file_pr)
U_pr_prr = row_normalize(M_pr_prr)
U_prr_rev = row_normalize(M_prr_rev)

M_file_dev = np.transpose(M_dev_file)
M_pr_file = np.transpose(M_file_pr)
M_prr_pr = np.transpose(M_pr_prr)
M_rev_prr = np.transpose(M_prr_rev)

U_file_dev = row_normalize(M_file_dev)
U_pr_file = row_normalize(M_pr_file)
U_prr_pr = row_normalize(M_prr_pr)
U_rev_prr = row_normalize(M_rev_prr)


### Questions

### 1

metapath1 = np.matmul(U_dev_file, np.matmul(U_file_pr, np.matmul(U_pr_prr,
U_prr_rev)))
metapath2 = np.matmul(U_rev_prr, np.matmul(U_prr_pr, np.matmul(U_pr_file,
U_file_dev)))


rank,_ = HRank_ASym(dev_names, rev_names, metapath1, metapath2)

print("Question 1: ")
print("Ranking of developers:", end=' ')
print(rank)


### 2

M_pr_pr = pr_accepted(project)

metapath1 = np.matmul(U_dev_file, np.matmul(U_file_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr, np.matmul(U_pr_prr,
U_prr_rev)))))
metapath2 = np.matmul(U_rev_prr, np.matmul(U_prr_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr, np.matmul(U_pr_file,
U_file_dev)))))


rank,_ = HRank_ASym(dev_names, rev_names, metapath1, metapath2)

print("Question 2: ")
print("Ranking of developers:", end=' ')
print(rank)


### 3

M_pr_pr = pr_rejected(project)

metapath1 = np.matmul(U_dev_file, np.matmul(U_file_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr,
np.matmul(U_pr_prr, U_prr_rev)))))
metapath2 = np.matmul(U_rev_prr, np.matmul(U_prr_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr,
np.matmul(U_pr_file,U_file_dev)))))

rank,_ = HRank_ASym(dev_names, rev_names, metapath1, metapath2)

print("Question 3: ")
print("Ranking of developers:", end=' ')
print(rank)


### 4

metapath1 = np.matmul(U_dev_file, np.matmul(U_file_pr, np.matmul(U_pr_prr,
U_prr_rev)))
metapath2 = np.matmul(U_rev_prr, np.matmul(U_prr_pr, np.matmul(U_pr_file,
U_file_dev)))

_, rank = HRank_ASym(dev_names, rev_names, metapath1, metapath2)

print("Question 4: ")
print("Ranking of reviewers:", end=' ')
print(rank)


### 5

M_pr_pr = pr_accepted(project)

metapath1 = np.matmul(U_dev_file, np.matmul(U_file_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr, np.matmul(U_pr_prr,
U_prr_rev)))))
metapath2 = np.matmul(U_rev_prr, np.matmul(U_prr_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr, np.matmul(U_pr_file,
U_file_dev)))))

_, rank = HRank_ASym(dev_names, rev_names, metapath1, metapath2)

print("Question 5: ")
print("Ranking of reviewers:", end=' ')
print(rank)


### 6

M_pr_pr = pr_rejected(project)

metapath1 = np.matmul(U_dev_file, np.matmul(U_file_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr,
np.matmul(U_pr_prr, U_prr_rev)))))
metapath2 = np.matmul(U_rev_prr, np.matmul(U_prr_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr,
np.matmul(U_pr_file,U_file_dev)))))

_, rank = HRank_ASym(dev_names, rev_names, metapath1, metapath2)

print("Question 6: ")
print("Ranking of reviewers:", end=' ')
print(rank)



### 7
decision = input("Enter decision (closed/merged/added_to_project) : ")
M_pr_pr = review_time_less_than_mean_time(project, decision)
metapath1 = np.matmul(U_dev_file, np.matmul(U_file_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr,
np.matmul(U_pr_prr, U_prr_rev)))))
metapath2 = np.matmul(U_rev_prr, np.matmul(U_prr_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr,
np.matmul(U_pr_file,U_file_dev)))))

rank,_ = HRank_ASym(dev_names, rev_names, metapath1, metapath2)

print("Question 7: ")
print("Ranking of developers:", end=' ')
print(rank)


### 8

M_pr_pr = review_time_more_than_mean_time(project, decision)
metapath1 = np.matmul(U_dev_file, np.matmul(U_file_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr,
np.matmul(U_pr_prr, U_prr_rev)))))
metapath2 = np.matmul(U_rev_prr, np.matmul(U_prr_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr,
np.matmul(U_pr_file,U_file_dev)))))

rank,_ = HRank_ASym(dev_names, rev_names, metapath1, metapath2)

print("Question 8: ")
print("Ranking of developers:", end=' ')
print(rank)


### 9

M_pr_pr = review_time_less_than_mean_time_accepted(project)
metapath1 = np.matmul(U_dev_file, np.matmul(U_file_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr,
np.matmul(U_pr_prr, U_prr_rev)))))
metapath2 = np.matmul(U_rev_prr, np.matmul(U_prr_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr,
np.matmul(U_pr_file,U_file_dev)))))

rank,_ = HRank_ASym(dev_names, rev_names, metapath1, metapath2)

print("Question 9: ")
print("Ranking of developers:", end=' ')
print(rank)


### 10

M_pr_pr = review_time_more_than_mean_time_accepted(project)
metapath1 = np.matmul(U_dev_file, np.matmul(U_file_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr,
np.matmul(U_pr_prr, U_prr_rev)))))
metapath2 = np.matmul(U_rev_prr, np.matmul(U_prr_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr,
np.matmul(U_pr_file,U_file_dev)))))


rank,_ = HRank_ASym(dev_names, rev_names, metapath1, metapath2)

print("Question 10: ")
print("Ranking of developers:", end=' ')
print(rank)