import numpy as np
from pymongo import MongoClient
import functools
def row_normalize(M):
    U = M
    for i in range(U.shape[0]):
        rowsum = U[i].sum()
        if rowsum > 0:
            U[i] = U[i] / rowsum
    return U


def custom_sort(a, b):
    if a[0]>b[0]:
        return 1
    elif a[0]==b[0] and a[1]<b[1]:
        return 1

    return -1


# A->P<-A
def HRank_Sym(A, M_p):
    # Restart vector
    E_restart = np.full((1, len(A)), 1/len(A))

    # Intial rank
    Vis_Prob = E_restart

    # alpha
    alpha = 0.15

    # HRank iterations
    prev_iter = np.full((1, len(A)), 1)
    cn = 0
    while True:
        Vis_Prob = alpha*np.matmul(Vis_Prob,M_p)+(1-alpha)*E_restart

        diff = np.max(np.absolute(prev_iter-Vis_Prob))
        if diff<0.0001:
            break
        prev_iter = Vis_Prob
        cn += 1

    rank = []
    for i in range(len(A)):
        rank.append([Vis_Prob[0][i], A[i]])

    rank.sort(key=functools.cmp_to_key(custom_sort))
    rank = [x[1] for x in rank]

    return rank

# metapath1 = np.matmul(U_dev_pr, np.matmul(U_pr_prr, U_prr_rev))         #dev*rev
# metapath2 = np.matmul(U_rev_prr, np.matmul(U_prr_pr, U_pr_dev))         #rev*dev

def HRank_ASym(A, B, M_p1, M_p2):
    # Restart vector
    E_restart1 = np.full((1, len(A)), 1/len(A))
    E_restart2 = np.full((1, len(B)), 1/len(B))

    # Intial rank
    Vis_Prob1 = E_restart1      #1*dev
    Vis_Prob2 = E_restart2      #1*rev

    # alpha
    alpha = 0.15

    # HRank iterations
    prev_iter1 = np.full((1, len(A)), 1)
    prev_iter2 = np.full((1, len(B)), 1)
    cn = 0
    while True:
        Vis_Prob1_copy = Vis_Prob1.copy()
        Vis_Prob2_copy = Vis_Prob2.copy()
        Vis_Prob1 = alpha*np.matmul(Vis_Prob2_copy,M_p2)+(1-alpha)*E_restart1
        Vis_Prob2 = alpha*np.matmul(Vis_Prob1_copy, M_p1)+(1-alpha)*E_restart2

        diff = max(np.max(np.absolute(prev_iter1-Vis_Prob1)), np.max(np.absolute(prev_iter2-Vis_Prob2)))
        if diff<0.0001:
            break
        prev_iter1 = Vis_Prob1
        prev_iter2 = Vis_Prob2
        cn += 1

    rank1 = []
    for i in range(len(A)):
        rank1.append([Vis_Prob1[0][i], A[i]])

    rank2 = []
    for i in range(len(B)):
        rank2.append([Vis_Prob1[0][i], B[i]])

    rank1.sort(key=functools.cmp_to_key(custom_sort))
    rank2.sort(key=functools.cmp_to_key(custom_sort))

    rank1 = [x[1] for x in rank1]
    rank2 = [x[1] for x in rank2]

    return rank1, rank2

# _, rank = HRank_ASym(devs, reviewers, metapath1, metapath2)

# ans = []
# for x in rank:
#     person = list(db["people"].find({"_id":x}, {"name":1, "_id":0}))[0]["name"]
#     ans.append(person)

# print(ans)

def mean_time_of_decision(project,decision):
    db = MongoClient("mongodb://localhost:27017/")["smartshark"]
    project_id = list(db["project"].find({"name" : project}, {"_id" : 1}))[0]["_id"]
    # print("Project ID: "+str(project_id))
    pull_request_system_id = list(db["pull_request_system"].find({"project_id":project_id}, {"_id": 1}))[0]["_id"]
    pr_ids=[x["_id"] for x in list(db["pull_request"].find({"pull_request_system_id":pull_request_system_id}, {"_id":1}))]
    pr_event_ids=dict()
    for p in pr_ids:
        # print(p)
        try:
            pr_event_ids[(list(db["pull_request_event"].find({"pull_request_id":p,"event_type":decision}, {"_id":1}))[0]["_id"])]=p
        except:
            print("event_not_found")
    # print(pr_event_ids)
    # pr_ids = list(db["pull_request_event"].find({"state":"merged"}, {"pull_request_id":1,"_id":0}))
    # pr_ids=[x["pull_request_id"] for x in pr_ids]
    sum_time=0
    for p in pr_event_ids:
        t2=(list(db["pull_request_event"].find({"_id":p}, {"created_at":1,"_id":0}))[0]["created_at"])
        t1=(list(db["pull_request"].find({"_id":pr_event_ids[p]}, {"created_at":1,"_id":0}))[0]["created_at"])
        td=t2-t1
        sum_time+=(td.total_seconds())
        return sum_time/len(pr_event_ids)