# project - giraph
# D -- PR -- PRC -- D
from hrank import row_normalize, HRank_ASym, HRank_Sym, mean_time_of_decision
from question import pr_accepted, pr_rejected, review_time_less_than_mean_time_accepted, review_time_less_than_mean_time, review_time_more_than_mean_time, review_time_more_than_mean_time_accepted
from pymongo import MongoClient
import numpy as np

from Build_reverse_identity_dictionary import Build_reverse_identity_dictionary

class Metapath3:
    def __init__(self, collection, project_name):
        self.BRID = Build_reverse_identity_dictionary()
        self.BRID.reading_identity_and_people_and_building_reverse_identity_dictionary()
        self.db = MongoClient("mongodb://localhost:27017/")["smartshark"]


        self.project = project_name
        # Fetching pull requests of project giraph
        project_id = list(self.db["project"].find({"name" : project}, {"_id" : 1}))[0]["_id"]

        pull_request_system_id = list(self.db["pull_request_system"].find({"project_id":project_id}, {"_id": 1}))[0]["_id"]
        vcs_system_id = list(self.db["vcs_system"].find({"project_id":project_id}, {"_id": 1}))[0]["_id"]
        issue_system_id = list(self.db["issue_system"].find({"project_id":project_id}, {"_id":1}))[0]["_id"]



        # Fetching the developers of that project with pull request
        devs = list(self.db["pull_request"].find({"pull_request_system_id":pull_request_system_id}, {"creator_id":1, "_id":0}))
        devs = [x["creator_id"] for x in devs]
        devs = set(devs)              # removing duplicates

        devs2 = list(self.db["commit"].find({"vcs_system_id":vcs_system_id}, {"author_id":1, "committer_id":1}))
        devs2 = [x["author_id"] for x in devs2]+[x["committer_id"] for x in devs2]
        devs2 = set(devs2)
        devs = devs.union(devs2)

        devs2 = list(self.db["issue"].find({"issue_system_id":issue_system_id}, {"creator_id":1}))
        devs3 = []
        for x in devs2:
            if 'creator_id' in x:
                devs3.append(x['creator_id'])
        devs3 = set(devs3)
        devs = devs.union(devs3)
        devs = list(devs)
        dev_indexes = {k:v for v, k in enumerate(devs)}


        id_to_name = {}
        for x in devs:
            person = list(self.db["people"].find({"_id": x}, {"name": 1, "_id": 0}))[0]["name"]
            id_to_name[self.BRID.reverse_identity_dict[x]] = person

        self.dev_names = []
        identities = set()
        for x in devs:
            identities.add(self.BRID.reverse_identity_dict[x])

        identities = list(identities)
        for x in identities:
            self.dev_names.append(id_to_name[x])

        id_indexes = {k: v for v, k in enumerate(identities)}



        # Fetching PRs
        prs = list(self.db["pull_request"].find({"pull_request_system_id":pull_request_system_id}, {"_id":1}))
        prs = [x["_id"] for x in prs]
        pr_indexes = {k:v for v, k in enumerate(prs)}



        # Fetching files
        pr_c = []
        for pr in prs:
            com = list(self.db["pull_request_comment"].find({"pull_request_id":pr}, {"_id":1}))
            com = [x["_id"] for x in com]
            pr_c= pr_c+com

        pr_c = list(set(pr_c))
        prc_indexes = {k:v for v, k in enumerate(pr_c)}



        # dev->PR
        M_dev_pr = np.zeros((len(identities), len(prs)))
        for i in range(len(devs)):
            pr_by_dev = list(self.db["pull_request"].find({"pull_request_system_id":pull_request_system_id,
                                                      "creator_id": devs[i]}, {"_id":1}))
            pr_by_dev = [x["_id"] for x in pr_by_dev]
            ids = self.BRID.reverse_identity_dict[devs[i]]
            for x in pr_by_dev:
                M_dev_pr[id_indexes[ids]][pr_indexes[x]] = 1


        self.U_dev_pr = row_normalize(M_dev_pr)


        # PR->PRC
        M_pr_prc = np.zeros((len(prs), len(pr_c)))
        for i in range(len(prs)):
            comment_by_pr = list(self.db["pull_request_comment"].find({"pull_request_id": prs[i]}))
            comment_by_pr = [x["_id"] for x in comment_by_pr]
            for x in comment_by_pr:
                M_pr_prc[i][prc_indexes[x]] = 1

        self.U_pr_prc = row_normalize(M_pr_prc)


        # PRC->dev
        M_prc_dev = np.zeros((len(pr_c), len(identities)))
        for i in range(len(pr_c)):
            dev_of_prc = list(self.db["pull_request_comment"].find({"_id": pr_c[i]}))
            dev_of_prc = [x["author_id"] for x in dev_of_prc]

            for x in dev_of_prc:
                if x in dev_indexes:
                    ids = self.BRID.reverse_identity_dict[x]
                    M_prc_dev[i][id_indexes[ids]] = 1

        self.U_prc_dev = row_normalize(M_prc_dev)

        M_dev_prc = np.transpose(M_prc_dev)
        M_prc_pr = np.transpose(M_pr_prc)
        M_pr_dev = np.transpose(M_dev_pr)

        self.U_dev_prc = row_normalize(M_dev_prc)
        self.U_prc_pr = row_normalize(M_prc_pr)
        self.U_pr_dev = row_normalize(M_pr_dev)


### Questions


    ### 1

    def q1(self):
        metapath1 = np.matmul(self.U_dev_pr, np.matmul(self.U_pr_prc, self.U_prc_dev))
        metapath2 = np.matmul(self.U_dev_prc, np.matmul(self.U_prc_pr, self.U_pr_dev))

        rank,_ = HRank_ASym(self.dev_names, self.dev_names, metapath1, metapath2)

        return rank



    ### 2
    def q2(self):
        M_pr_pr = pr_accepted(self.project)

        metapath1 = np.matmul(self.U_dev_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr, np.matmul(self.U_pr_prc, self.U_prc_dev))))
        metapath2 = np.matmul(self.U_dev_prc, np.matmul(self.U_prc_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr, self.U_pr_dev))))

        rank,_ = HRank_ASym(self.dev_names, self.dev_names, metapath1, metapath2)

        return rank



    ### 3
    def q3(self):
        M_pr_pr = pr_rejected(self.project)
        metapath1 = np.matmul(self.U_dev_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr, np.matmul(self.U_pr_prc, self.U_prc_dev))))
        metapath2 = np.matmul(self.U_dev_prc, np.matmul(self.U_prc_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr, self.U_pr_dev))))

        rank,_ = HRank_ASym(self.dev_names, self.dev_names, metapath1, metapath2)

        return rank



    ### 7
    def q7(self):
        decision = input("Enter decision (closed/merged/added_to_project) : ")
        M_pr_pr = review_time_less_than_mean_time(self.project, decision)
        if M_pr_pr[0]=="Decision not found":
            return M_pr_pr[0]

        metapath1 = np.matmul(self.U_dev_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr, np.matmul(self.U_pr_prc, self.U_prc_dev))))
        metapath2 = np.matmul(self.U_dev_prc, np.matmul(self.U_prc_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr, self.U_pr_dev))))

        rank,_ = HRank_ASym(self.dev_names, self.dev_names, metapath1, metapath2)

        return rank



    ### 8
    def q8(self):
        decision = input("Enter decision (closed/merged/added_to_project) : ")
        M_pr_pr = review_time_more_than_mean_time(self.project, decision)
        if M_pr_pr[0]=="Decision not found":
            return M_pr_pr[0]

        metapath1 = np.matmul(self.U_dev_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr, np.matmul(self.U_pr_prc, self.U_prc_dev))))
        metapath2 = np.matmul(self.U_dev_prc, np.matmul(self.U_prc_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr, self.U_pr_dev))))

        rank,_ = HRank_ASym(self.dev_names, self.dev_names, metapath1, metapath2)

        return rank



    ### 9
    def q9(self):
        M_pr_pr = review_time_less_than_mean_time_accepted(project)
        if M_pr_pr[0]=="Decision not found":
            return M_pr_pr[0]

        metapath1 = np.matmul(self.U_dev_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr, np.matmul(self.U_pr_prc, self.U_prc_dev))))
        metapath2 = np.matmul(self.U_dev_prc, np.matmul(self.U_prc_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr, self.U_pr_dev))))

        rank,_ = HRank_ASym(self.dev_names, self.dev_names, metapath1, metapath2)

        return rank



    ### 10
    def q10(self):
        M_pr_pr = review_time_more_than_mean_time_accepted(project)
        if M_pr_pr[0]=="Decision not found":
            return M_pr_pr[0]

        metapath1 = np.matmul(self.U_dev_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr, np.matmul(self.U_pr_prc, self.U_prc_dev))))
        metapath2 = np.matmul(self.U_dev_prc, np.matmul(self.U_prc_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr, self.U_pr_dev))))

        rank,_ = HRank_ASym(self.dev_names, self.dev_names, metapath1, metapath2)

        return rank

if __name__ == "__main__":
    project = input("Enter project name: ")

    mp2 = Metapath3("smartshark", project)
    print("Question 1: ", end='')
    print(mp2.q1())

    print("Question 2: ", end='')
    print(mp2.q2())

    print("Question 3: ", end='')
    print(mp2.q3())

    print("Question 7: ", end='')
    print(mp2.q7())

    print("Question 8: ", end='')
    print(mp2.q8())

    print("Question 9: ", end='')
    print(mp2.q9())

    print("Question 10: ", end='')
    print(mp2.q10())