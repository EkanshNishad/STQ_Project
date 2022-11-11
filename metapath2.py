# D1 -> PR -> F <- PR <- D2
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
pr_indexes = {k:v for v,k in enumerate(prs)}



# Fetching files
files = list(db["file"].find({"vcs_system_id":vcs_system_id}, {"_id":1}))
files = [x['_id'] for x in files]
files = set(files)
file_indexes = {k:v for v, k in enumerate(files)}



# dev->PR and PR->dev
M_dev_pr = np.zeros((len(devs), len(prs)))
for i in range(len(devs)):
    pr_by_dev = list(db["pull_request"].find({"pull_request_system_id":pull_request_system_id,
                                              "creator_id": devs[i]}, {"_id":1}))
    pr_by_dev = [x["_id"] for x in pr_by_dev]
    for x in pr_by_dev:
        M_dev_pr[i][pr_indexes[x]] = 1


U_dev_pr = row_normalize(M_dev_pr)
M_pr_dev = np.transpose(M_dev_pr)
U_pr_dev = row_normalize(M_pr_dev)

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

U_pr_file = row_normalize(M_pr_file)
M_file_pr = np.transpose(M_pr_file)
U_file_pr = row_normalize(M_file_pr)



### Questions

### 1

metapath = np.matmul(U_dev_pr, np.matmul(U_pr_file, np.matmul(U_file_pr, U_pr_dev)))
metapath = row_normalize(metapath)
rank = HRank_Sym(dev_names, metapath)

print("Question 1: ")
print("Ranking of developers:", end=' ')
print(rank)



### 2
M_pr_pr = pr_accepted(project)

metapath = np.matmul(U_dev_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr, np.matmul(U_pr_file,
np.matmul(U_file_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr, U_pr_dev)))))))
metapath = row_normalize(metapath)

rank = HRank_Sym(dev_names, metapath)
print("Question 2: ")
print("Ranking of developers:", end=' ')
print(rank)



### 3
M_pr_pr = pr_rejected(project)

metapath = np.matmul(U_dev_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr, np.matmul(U_pr_file,
np.matmul(U_file_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr, U_pr_dev)))))))
metapath = row_normalize(metapath)

rank = HRank_Sym(dev_names, metapath)
print("Question 3: ")
print("Ranking of developers:", end=' ')
print(rank)



### 7
decision = input("Enter decision (closed/merged/added_to_project) : ")
M_pr_pr = review_time_less_than_mean_time(project, decision)

metapath = np.matmul(U_dev_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr, np.matmul(U_pr_file,
np.matmul(U_file_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr, U_pr_dev)))))))
metapath = row_normalize(metapath)

rank = HRank_Sym(dev_names, metapath)

print("Question 7: ")
print("Ranking of developers:", end=' ')
print(rank)



### 8

M_pr_pr = review_time_more_than_mean_time(project, decision)
metapath = np.matmul(U_dev_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr, np.matmul(U_pr_file,
np.matmul(U_file_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr, U_pr_dev)))))))
metapath = row_normalize(metapath)

rank = HRank_Sym(dev_names, metapath)


print("Question 8: ")
print("Ranking of developers:", end=' ')
print(rank)



### 9

M_pr_pr = review_time_less_than_mean_time_accepted(project)
metapath = np.matmul(U_dev_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr, np.matmul(U_pr_file,
np.matmul(U_file_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr, U_pr_dev)))))))
metapath = row_normalize(metapath)

rank = HRank_Sym(dev_names, metapath)

print("Question 9: ")
print("Ranking of developers:", end=' ')
print(rank)



### 10

M_pr_pr = review_time_more_than_mean_time_accepted(project)
metapath = np.matmul(U_dev_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr, np.matmul(U_pr_file,
np.matmul(U_file_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr, U_pr_dev)))))))
metapath = row_normalize(metapath)

rank = HRank_Sym(dev_names, metapath)

print("Question 10: ")
print("Ranking of developers:", end=' ')
print(rank)



