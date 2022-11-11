# D -- F -- PR -- I
import sys

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


# Fetching files
files = list(db["file"].find({"vcs_system_id":vcs_system_id}, {"_id":1}))
files = [x['_id'] for x in files]
files = set(files)
file_indexes = {k:v for v, k in enumerate(files)}


# Fetching PRs
prs = list(db["pull_request"].find({"pull_request_system_id":pull_request_system_id}, {"_id":1}))
prs = [x["_id"] for x in prs]
pr_indexes = {k:v for v, k in enumerate(prs)}


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


# dev<->file
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


# file<->pr
for i in range(len(prs)):
    file_by_pr = list(db["pull_request_file"].find({"pull_request_id":prs[i]}, {"path":1}))
    file_by_pr = [x['path'] for x in file_by_pr]
    for pf in file_by_pr:
        if pf in path_to_file:
            M_file_pr[path_to_file[pf]][i] = 1


# pr<->int
M_pr_int = np.zeros((len(prs), len(integrators)))
for i in range(len(prs)):
    int_by_pr = list(db["pull_request_event"].find({"pull_request_id":prs[i], "event_type":"merged"},
                                                   {"pull_request_id":1, "author_id":1}))
    for x in int_by_pr:
        if x['author_id'] in inte_indexes:
            M_pr_int[i][inte_indexes[x['author_id']]] = 1

U_dev_file = row_normalize(M_dev_file)
U_file_pr = row_normalize(M_file_pr)
U_pr_int = row_normalize(M_pr_int)

M_int_pr = np.transpose(M_pr_int)
M_pr_file = np.transpose(M_file_pr)
M_file_dev = np.transpose(M_dev_file)

U_file_dev = row_normalize(M_file_dev)
U_pr_file = row_normalize(M_pr_file)
U_int_pr = row_normalize(M_int_pr)


### Questions

### 1

metapath1 = np.matmul(U_dev_file, np.matmul(U_file_pr, U_pr_int))
metapath2 = np.matmul(U_int_pr, np.matmul(U_pr_file, U_file_dev))

rank, _ = HRank_ASym(dev_names, inte_names, metapath1, metapath2)

print("Question 1: ")
print("Ranking of developers:", end=' ')
print(rank)



### 2
M_pr_pr = pr_accepted(project)
metapath1 = np.matmul(U_dev_file, np.matmul(U_file_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr,
U_pr_int))))
metapath2 = np.matmul(U_int_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr, np.matmul(U_pr_file,
U_file_dev))))

rank, _ = HRank_ASym(dev_names, inte_names, metapath1, metapath2)

print("Question 2: ")
print("Ranking of developers:", end=' ')
print(rank)



### 3
M_pr_pr = pr_rejected(project)

metapath1 = np.matmul(U_dev_file, np.matmul(U_file_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr,
U_pr_int))))
metapath2 = np.matmul(U_int_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr, np.matmul(U_pr_file,
U_file_dev))))

rank,_ = HRank_ASym(dev_names, inte_names, metapath1, metapath2)

print("Question 3: ")
print("Ranking of developers:", end=' ')
print(rank)



### 7
decision = input("Enter decision (closed/merged/added_to_project) : ")
M_pr_pr = review_time_less_than_mean_time(project, decision)
metapath1 = np.matmul(U_dev_file, np.matmul(U_file_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr,
U_pr_int))))
metapath2 = np.matmul(U_int_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr, np.matmul(U_pr_file,
U_file_dev))))

rank,_ = HRank_ASym(dev_names, inte_names, metapath1, metapath2)

print("Question 7: ")
print("Ranking of developers:", end=' ')
print(rank)


### 8

M_pr_pr = review_time_more_than_mean_time(project, decision)
metapath1 = np.matmul(U_dev_file, np.matmul(U_file_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr,
U_pr_int))))
metapath2 = np.matmul(U_int_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr, np.matmul(U_pr_file,
U_file_dev))))

rank,_ = HRank_ASym(dev_names, inte_names, metapath1, metapath2)

print("Question 8: ")
print("Ranking of developers:", end=' ')
print(rank)


### 9

M_pr_pr = review_time_less_than_mean_time_accepted(project)
metapath1 = np.matmul(U_dev_file, np.matmul(U_file_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr,
U_pr_int))))
metapath2 = np.matmul(U_int_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr, np.matmul(U_pr_file,
U_file_dev))))

rank,_ = HRank_ASym(dev_names, inte_names, metapath1, metapath2)

print("Question 9: ")
print("Ranking of developers:", end=' ')
print(rank)


### 10

M_pr_pr = review_time_more_than_mean_time_accepted(project)
metapath1 = np.matmul(U_dev_file, np.matmul(U_file_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr,
U_pr_int))))
metapath2 = np.matmul(U_int_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr, np.matmul(U_pr_file,
U_file_dev))))


rank,_ = HRank_ASym(dev_names, inte_names, metapath1, metapath2)

print("Question 10: ")
print("Ranking of developers:", end=' ')
print(rank)