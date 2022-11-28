# D -- PR -- PRreview -- R

from hrank import row_normalize, HRank_ASym, HRank_Sym, mean_time_of_decision
from question import pr_accepted, pr_rejected, review_time_less_than_mean_time_accepted, review_time_less_than_mean_time, review_time_more_than_mean_time, review_time_more_than_mean_time_accepted
from pymongo import MongoClient
import numpy as np

from Build_reverse_identity_dictionary import Build_reverse_identity_dictionary

class Metapath6:
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



        # Fetching PR reviews and reviewers
        prrevs = set()
        reviewers = set()
        for pr in prs:
            reviews = list(self.db["pull_request_review"].find({"pull_request_id":pr}, {"_id":1, "creator_id":1}))
            prr = set([x['_id'] for x in reviews])
            rev = set()
            for x in reviews:
                if 'creator_id' in x:
                    ids = self.BRID.reverse_identity_dict[x['creator_id']]
                    person = list(self.db["people"].find({"_id": x['creator_id']}, {"name": 1, "_id": 0}))[0]["name"]
                    id_to_name[ids] = person
                    rev.add(ids)
            prrevs = prrevs.union(prr)
            reviewers = reviewers.union(rev)

        prrevs = list(prrevs)
        reviewers = list(reviewers)
        prr_indexes = {k:v for v,k in enumerate(prrevs)}
        rev_indexes = {k:v for v,k in enumerate(reviewers)}

        self.rev_names = []
        for x in reviewers:
            self.rev_names.append(id_to_name[x])


        # Creating matrix for metapath
        M_dev_pr = np.zeros((len(identities), len(prs)))
        M_pr_prr = np.zeros((len(prs), len(prrevs)))
        M_prr_rev = np.zeros((len(prrevs), len(reviewers)))


        # dev->PR
        for i in range(len(devs)):
            pr_by_dev = list(self.db["pull_request"].find({"pull_request_system_id":pull_request_system_id,
                                                      "creator_id": devs[i]}, {"_id:1"}))
            pr_by_dev = [x["_id"] for x in pr_by_dev]
            ids = self.BRID.reverse_identity_dict[devs[i]]
            for x in pr_by_dev:
                M_dev_pr[id_indexes[ids]][pr_indexes[x]] = 1



        #PR->PRreview->Reviewers
        for i in range(len(prs)):
            rev_pr = list(self.db["pull_request_review"].find({"pull_request_id":prs[i]}, {"creator_id":1, "_id":1}))
            for x in rev_pr:
                M_pr_prr[i][prr_indexes[x['_id']]] = 1
            for x in rev_pr:
                ids = self.BRID.reverse_identity_dict[x['creator_id']]
                M_prr_rev[prr_indexes[x['_id']]][rev_indexes[ids]] = 1


        self.U_dev_pr = row_normalize(M_dev_pr)
        self.U_pr_prr = row_normalize(M_pr_prr)
        self.U_prr_rev = row_normalize(M_prr_rev)


        M_pr_dev = np.transpose(M_dev_pr)
        M_prr_pr = np.transpose(M_pr_prr)
        M_rev_prr = np.transpose(M_prr_rev)

        self.U_pr_dev = row_normalize(M_pr_dev)
        self.U_prr_pr = row_normalize(M_prr_pr)
        self.U_rev_prr = row_normalize(M_rev_prr)


    ### Questions


    ### 1
    def q1(self):
        metapath1 = np.matmul(self.U_dev_pr, np.matmul(self.U_pr_prr, self.U_prr_rev))
        metapath2 = np.matmul(self.U_rev_prr, np.matmul(self.U_prr_pr, self.U_pr_dev))
        rank,_ = HRank_ASym(self.dev_names, self.rev_names, metapath1, metapath2)

        return rank



    ### 2
    def q2(self):
        M_pr_pr = pr_accepted(self.project)

        metapath1 = np.matmul(self.U_dev_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr, np.matmul(self.U_pr_prr, self.U_prr_rev))))
        metapath2 = np.matmul(self.U_rev_prr, np.matmul(self.U_prr_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr, self.U_pr_dev))))

        rank,_ = HRank_ASym(self.dev_names, self.rev_names, metapath1, metapath2)

        return rank



    ### 3
    def q3(self):
        M_pr_pr = pr_rejected(self.project)

        metapath1 = np.matmul(self.U_dev_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr, np.matmul(self.U_pr_prr, self.U_prr_rev))))
        metapath2 = np.matmul(self.U_rev_prr, np.matmul(self.U_prr_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr, self.U_pr_dev))))

        rank,_ = HRank_ASym(self.dev_names, self.rev_names, metapath1, metapath2)
        return rank



    ### 4
    def q4(self):
        metapath1 = np.matmul(self.U_dev_pr, np.matmul(self.U_pr_prr, self.U_prr_rev))
        metapath2 = np.matmul(self.U_rev_prr, np.matmul(self.U_prr_pr, self.U_pr_dev))

        _, rank = HRank_ASym(self.dev_names, self.rev_names, metapath1, metapath2)

        return rank



    ### 5
    def q5(self):
        M_pr_pr = pr_accepted(self.project)

        metapath1 = np.matmul(self.U_dev_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr, np.matmul(self.U_pr_prr, self.U_prr_rev))))
        metapath2 = np.matmul(self.U_rev_prr, np.matmul(self.U_prr_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr, self.U_pr_dev))))

        _, rank = HRank_ASym(self.dev_names, self.rev_names, metapath1, metapath2)

        return rank



    ### 6
    def q6(self):
        M_pr_pr = pr_rejected(self.project)

        metapath1 = np.matmul(self.U_dev_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr, np.matmul(self.U_pr_prr, self.U_prr_rev))))
        metapath2 = np.matmul(self.U_rev_prr, np.matmul(self.U_prr_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr, self.U_pr_dev))))

        _, rank = HRank_ASym(self.dev_names, self.rev_names, metapath1, metapath2)

        return rank



    ### 7
    def q7(self):
        decision = input("Enter decision (closed/merged/added_to_project) : ")
        M_pr_pr = review_time_less_than_mean_time(self.project, decision)
        if M_pr_pr[0] == "Decision not found":
            return M_pr_pr[0]

        metapath1 = np.matmul(self.U_dev_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr, np.matmul(self.U_pr_prr, self.U_prr_rev))))
        metapath2 = np.matmul(self.U_rev_prr, np.matmul(self.U_prr_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr, self.U_pr_dev))))

        rank,_ = HRank_ASym(self.dev_names, self.rev_names, metapath1, metapath2)

        return rank



    ### 8
    def q8(self):
        decision = input("Enter decision (closed/merged/added_to_project) : ")
        M_pr_pr = review_time_more_than_mean_time(project, decision)
        if M_pr_pr[0] == "Decision not found":
            return M_pr_pr[0]

        metapath1 = np.matmul(self.U_dev_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr, np.matmul(self.U_pr_prr, self.U_prr_rev))))
        metapath2 = np.matmul(self.U_rev_prr, np.matmul(self.U_prr_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr, self.U_pr_dev))))

        rank,_ = HRank_ASym(self.dev_names, self.rev_names, metapath1, metapath2)

        return rank



    ### 9
    def q9(self):
        M_pr_pr = review_time_less_than_mean_time_accepted(project)

        metapath1 = np.matmul(self.U_dev_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr, np.matmul(self.U_pr_prr, self.U_prr_rev))))
        metapath2 = np.matmul(self.U_rev_prr, np.matmul(self.U_prr_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr, self.U_pr_dev))))

        rank,_ = HRank_ASym(self.dev_names, self.rev_names, metapath1, metapath2)

        return rank



    ### 10
    def q10(self):
        M_pr_pr = review_time_more_than_mean_time_accepted(project)

        metapath1 = np.matmul(self.U_dev_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr, np.matmul(self.U_pr_prr, self.U_prr_rev))))
        metapath2 = np.matmul(self.U_rev_prr, np.matmul(self.U_prr_pr, np.matmul(M_pr_pr, np.matmul(M_pr_pr, self.U_pr_dev))))

        rank,_ = HRank_ASym(self.dev_names, self.rev_names, metapath1, metapath2)

        return rank

if __name__ == "__main__":
    project = input("Enter project name: ")

    mp2 = Metapath6("smartshark", project)
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