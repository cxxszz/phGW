import numpy as np
import os
import pickle
import json
import ast
import gc
from copy import deepcopy as dc
from scipy.sparse import csr_matrix, save_npz, load_npz
from data_io.real_noise_hete import graph, graph_pair_rn
from data_io.part import part_static, part_dynamic, subgraph, subgraph_type, part_dynamic_type
from sortedcontainers import SortedDict, SortedList
from lib.util import get_nei

data_fold = "/home/lwj/dataset/oag/"
# ======================================== reducing storage ==================================
raw_oag_fold = os.path.join(data_fold, "raw_oag")


# ---------------------------- venue nodes do not require large storages ---------------------
def author_nodes_pre():
    # file_name = "aminer_authors_0.txt"
    # file_name = "aminer_authors_1.txt"
    # file_name = "aminer_authors_2.txt"
    # file_name = "aminer_authors_3.txt"
    # file_name = "aminer_authors_4.txt"
    # big_path = os.path.join(data_fold, "aminer_authors_0", file_name)
    # file_name = "aminer_authors_5.txt"
    # file_name = "aminer_authors_6.txt"
    # file_name = "aminer_authors_7.txt"
    # file_name = "aminer_authors_8.txt"
    # file_name = "aminer_authors_9.txt"
    # big_path = os.path.join(data_fold, "aminer_authors_1", file_name)
    # file_name = "aminer_authors_10.txt"
    # file_name = "aminer_authors_11.txt"
    # file_name = "aminer_authors_12.txt"
    # file_name = "aminer_authors_13.txt"
    # file_name = "aminer_authors_14.txt"
    # big_path = os.path.join(data_fold, "aminer_authors_2", file_name)
    # file_name = "aminer_authors_15.txt"
    # file_name = "aminer_authors_16.txt"
    # file_name = "aminer_authors_17.txt"
    # file_name = "aminer_authors_18.txt"
    # file_name = "aminer_authors_19.txt"
    # big_path = os.path.join(data_fold, "aminer_authors_3", file_name)

    # ----------------------------- mag ---------------------------------
    # file_name = "mag_authors_0.txt"
    # file_name = "mag_authors_1.txt"
    # file_name = "mag_authors_2.txt"
    # file_name = "mag_authors_3.txt"
    # file_name = "mag_authors_4.txt"
    # big_path = os.path.join(data_fold, "mag_authors_0", file_name)
    # file_name = "mag_authors_5.txt"
    # file_name = "mag_authors_6.txt"
    # file_name = "mag_authors_7.txt"
    # file_name = "mag_authors_8.txt"
    # file_name = "mag_authors_9.txt"
    # big_path = os.path.join(data_fold, "mag_authors_1", file_name)
    # file_name = "mag_authors_10.txt"
    # file_name = "mag_authors_11.txt"
    file_name = "mag_authors_12.txt"
    big_path = os.path.join(data_fold, "mag_authors_2", file_name)
    small_path = os.path.join(raw_oag_fold, file_name)
    small_file = open(small_path, "w")
    for line in open(big_path, "r"):
        big_d = json.loads(line)
        small_d = {}
        small_d["id"] = big_d["id"]
        try:
            small_d["name"] = big_d["name"]
        except KeyError:
            pass
        print(small_d)
        small_file.write(str(small_d) + "\n")
    small_file.close()
    return 0


def paper_nodes_pre():
    # file_name = "aminer_papers_0.txt"
    # file_name = "aminer_papers_1.txt"
    # file_name = "aminer_papers_2.txt"
    # file_name = "aminer_papers_3.txt"
    # big_path = os.path.join(data_fold, "aminer_papers_0", file_name)
    # file_name = "aminer_papers_4.txt"
    # file_name = "aminer_papers_5.txt"
    # file_name = "aminer_papers_6.txt"
    # file_name = "aminer_papers_7.txt"
    # big_path = os.path.join(data_fold, "aminer_papers_1", file_name)
    # file_name = "aminer_papers_8.txt"
    # file_name = "aminer_papers_9.txt"
    # file_name = "aminer_papers_10.txt"
    # file_name = "aminer_papers_11.txt"
    # big_path = os.path.join(data_fold, "aminer_papers_2", file_name)
    # file_name = "aminer_papers_12.txt"
    # file_name = "aminer_papers_13.txt"
    # file_name = "aminer_papers_14.txt"
    # big_path = os.path.join(data_fold, "aminer_papers_3", file_name)

    # --------------------------- mag ---------------------------------
    file_name = "mag_papers_0.txt"
    # file_name = "mag_papers_1.txt"
    # file_name = "mag_papers_2.txt"
    # file_name = "mag_papers_3.txt"
    big_path = os.path.join(data_fold, "mag_papers_0", file_name)
    # file_name = "mag_papers_4.txt"
    # file_name = "mag_papers_5.txt"
    # file_name = "mag_papers_6.txt"
    # file_name = "mag_papers_7.txt"
    # big_path = os.path.join(data_fold, "mag_papers_1", file_name)
    # file_name = "mag_papers_8.txt"
    # file_name = "mag_papers_9.txt"
    # file_name = "mag_papers_10.txt"
    # big_path = os.path.join(data_fold, "mag_papers_2", file_name)
    small_path = os.path.join(raw_oag_fold, file_name)
    small_file = open(small_path, "w")
    for line in open(big_path, "r"):
        big_d = json.loads(line)
        small_d = {}
        small_d["id"] = big_d["id"]
        try:
            small_d["title"] = big_d["title"]
        except KeyError:
            pass
        small_d["authors"] = []
        for d in big_d["authors"]:
            print("d={}".format(d))
            try:
                small_d["authors"].append(d["id"])
            except KeyError:
                pass

        try:
            print('big_d["venue"]={}'.format(big_d["venue"]))
            small_d["venue"] = big_d["venue"]["id"]
        except KeyError:
            pass
        print("small_d={}".format(small_d))
        small_file.write(str(small_d) + "\n")
    small_file.close()
    return 0


def char2num(c: str):
    if ord(c) >= ord('a') and ord(c) <= ord('z'):
        return ord(c) - ord('a')
    elif ord(c) >= ord('A') and ord(c) <= ord('Z'):
        return ord(c) - ord('A')
    else:
        return 26 + ord(c) % 6


def count(name: str):
    hist = np.zeros(32)
    for c in name:
        index = char2num(c)
        hist[index] += 1
    return hist


author_type = 0
paper_type = 1
venue_type = 2


def display(indices: list, node_types: list):
    n_author = 0
    n_paper = 0
    n_venue = 0
    for i in indices:
        if node_types[i] == author_type:
            n_author += 1
        elif node_types[i] == paper_type:
            n_paper += 1
        else:
            n_venue += 1
    print("author_nodes={}, paper_nodes={}, venue_nodes={}".format(n_author, n_paper, n_venue))
    return 0


def overlapped(index: int, node_ids: SortedList, links: SortedDict):
    str_id = node_ids[index]
    if str_id in links.keys():
        return True
    else:
        return False


add_percent = 0.9


def reasonable_size(w_csr: csr_matrix, node_types: list, node_ids: SortedList, links: SortedDict, min_size=4000,
                    max_size=10000, ini_node=0):
    indices = [ini_node]
    used = []
    while True:
        for i in indices:
            if i not in used:
                nei = get_nei(w=w_csr, i=i)
                for j in nei:
                    if j not in indices:
                        if overlapped(index=j, node_ids=node_ids, links=links):
                            indices.append(j)
                        else:
                            if np.random.uniform(0, 1) > add_percent:  # include more overlapped nodes
                                indices.append(j)
                        if len(indices) > min_size:
                            break
            used.append(i)
            if len(indices) > min_size:
                break
        if len(indices) > min_size:
            display(indices=indices, node_types=node_types)
            break
    assert len(indices) < max_size
    return indices


def main_api_type(seed=1, min_size=80000, max_size=110000):
    all_paths = os.listdir(raw_oag_fold)
    # ================================== linking ==================================
    null_tag = "NoCP_Tag_added_by_lwj"
    link_file_path = os.path.join(raw_oag_fold, "link.pt")
    if os.path.exists(link_file_path):
        print("loading saved links")
        with open(link_file_path, "rb") as f:
            links = pickle.load(f)
    else:
        print("obtaining links")
        link_names = ["author_linking_pairs.txt", "paper_linking_pairs.txt", "venue_linking_pairs.txt"]
        link_paths = [os.path.join(raw_oag_fold, name) for name in link_names]
        links = SortedDict()
        for link_path in link_paths:
            for line in open(link_path, "r"):
                tmp_d = ast.literal_eval(line)
                links[tmp_d["mid"]] = tmp_d["aid"]
        print("len(links)={}".format(len(links)))
        with open(link_file_path, "wb") as f:
            pickle.dump(links, f)
    # =================================== mag =====================================
    print("\n=================== processing the MAG graph =====================\n")
    mag_paths = [os.path.join(raw_oag_fold, path) for path in all_paths if "mag" in path]
    mag_author_paths = [path for path in mag_paths if "author" in path]
    mag_paper_paths = [path for path in mag_paths if "paper" in path]
    mag_venue_paths = [path for path in mag_paths if "venue" in path]
    # --------------------- obtain the index of each node ----------------------
    id_mag_path = os.path.join(raw_oag_fold, "id_mag.pt")
    if os.path.exists(id_mag_path):
        print("loading the saved indices")
        with open(id_mag_path, "rb") as f:
            id_mag = pickle.load(f)
    else:
        print("obtaining the index of each node")
        id_mag = SortedList()
        for path in mag_paths:
            print("path={}".format(path))
            for line in open(path, "r"):
                tmp_d = ast.literal_eval(line)
                id_mag.add(tmp_d["id"])
        with open(id_mag_path, "wb") as f:
            pickle.dump(id_mag, f)
    Ns = len(id_mag)
    # --------------------- obtain the underlying identities -----------------------
    lying_s_full_path = os.path.join(raw_oag_fold, "lying_s_full.pt")
    if os.path.exists(lying_s_full_path):
        print("loading the underlying identities")
        with open(lying_s_full_path, "rb") as f:
            lying_s_full = pickle.load(f)
    else:
        print("obtaining the underlying identities")
        lying_s_full = []
        for mid in id_mag:
            try:
                lying_s_full.append(links[mid])
            except KeyError:
                lying_s_full.append(null_tag)
        with open(lying_s_full_path, "wb") as f:
            pickle.dump(lying_s_full, f)
    # --------------------------- obtain the type of each node ----------------------
    node_types_s_full_path = os.path.join(raw_oag_fold, "node_types_s_full.npy")
    if os.path.exists(node_types_s_full_path):
        print("loading node_types_s")
        node_types_s_full = np.load(node_types_s_full_path)
    else:
        print("obtaining the node_types_s_full")
        node_types_s_full = np.zeros(Ns, dtype=np.int)
        for path in mag_author_paths:
            print("path={}".format(path))
            for line in open(path, "r"):
                tmp_d = ast.literal_eval(line)
                index = id_mag.bisect_left(tmp_d["id"])
                node_types_s_full[index] = author_type
        for path in mag_paper_paths:
            print("path={}".format(path))
            for line in open(path, "r"):
                tmp_d = ast.literal_eval(line)
                index = id_mag.bisect_left(tmp_d["id"])
                node_types_s_full[index] = paper_type
        for path in mag_venue_paths:
            print("path={}".format(path))
            for line in open(path, "r"):
                tmp_d = ast.literal_eval(line)
                index = id_mag.bisect_left(tmp_d["id"])
                node_types_s_full[index] = venue_type
        np.save(node_types_s_full_path, node_types_s_full)

    # --------------------------- obtain values and adjacencies ---------------------
    values_s_full_path = os.path.join(raw_oag_fold, "values_s_full.npy")
    adj_s_full_path = os.path.join(raw_oag_fold, "adj_s_full.npz")
    if os.path.exists(values_s_full_path) and os.path.exists(adj_s_full_path):
        print("loading values and adjacencies")
        values_s_full = np.load(values_s_full_path)
        adj_s_full = load_npz(adj_s_full_path)
    else:
        print("obtaining values and adjacencies")
        # ------------------ mistakes in papers files ---------------------------
        print("obtaining reversed venue IDs")
        r_venue_id = SortedDict()
        for path in mag_venue_paths:
            print("path={}".format(path))
            for line in open(path, "r"):
                tmp_d = ast.literal_eval(line)
                if "JournalId" in tmp_d.keys():
                    r_venue_id[tmp_d["JournalId"]] = tmp_d["id"]
                else:
                    r_venue_id[tmp_d["ConferenceId"]] = tmp_d["id"]
        values_s_full = np.zeros([Ns, 32])
        for path in mag_author_paths:
            print("path={}".format(path))
            for line in open(path, "r"):
                tmp_d = ast.literal_eval(line)
                index = id_mag.bisect_left(tmp_d["id"])
                try:
                    values_s_full[index] = count(tmp_d["name"])
                except KeyError:
                    pass
        for path in mag_venue_paths:
            print("path={}".format(path))
            for line in open(path, "r"):
                tmp_d = ast.literal_eval(line)
                index = id_mag.bisect_left(tmp_d["id"])
                try:
                    values_s_full[index] = count(tmp_d["DisplayName"])
                except KeyError:
                    pass
        weights, rows, cols = [], [], []
        for path in mag_paper_paths:
            print("path={}".format(path))
            for line in open(path, "r"):
                tmp_d = ast.literal_eval(line)
                i = id_mag.bisect_left(tmp_d["id"])
                try:
                    values_s_full[i] = count(tmp_d["title"])
                except KeyError:
                    pass
                # edges between paper nodes and author nodes
                for author_id in tmp_d["authors"]:
                    j = id_mag.bisect_left(author_id)
                    if j < Ns:
                        weights.append(1)
                        weights.append(1)
                        rows.append(i)
                        rows.append(j)
                        cols.append(j)
                        cols.append(i)
                try:
                    venue_id = r_venue_id[tmp_d["venue"]]
                    j = id_mag.bisect_left(venue_id)
                    if j < Ns:
                        weights.append(1)
                        weights.append(1)
                        rows.append(i)
                        rows.append(j)
                        cols.append(j)
                        cols.append(i)
                except KeyError:
                    pass
        adj_s_full = csr_matrix((weights, (rows, cols)), shape=(Ns, Ns))
        np.save(values_s_full_path, values_s_full)
        save_npz(adj_s_full_path, adj_s_full)
    # ================================== aminer ===================================
    print("\n=================== processing the Aminer graph =====================\n")
    aminer_paths = [os.path.join(raw_oag_fold, path) for path in all_paths if "aminer" in path]
    aminer_author_paths = [path for path in aminer_paths if "author" in path]
    aminer_paper_paths = [path for path in aminer_paths if "paper" in path]
    aminer_venue_paths = [path for path in aminer_paths if "venue" in path]
    # --------------------- obtain the index of each node ----------------------
    id_aminer_path = os.path.join(raw_oag_fold, "id_aminer.pt")
    if os.path.exists(id_aminer_path):
        print("loading the saved indices")
        with open(id_aminer_path, "rb") as f:
            id_aminer = pickle.load(f)
    else:
        print("obtaining the index of each node")
        id_aminer = SortedList()
        for path in aminer_paths:
            print("path={}".format(path))
            for line in open(path, "r"):
                tmp_d = ast.literal_eval(line)
                id_aminer.add(tmp_d["id"])
        with open(id_aminer_path, "wb") as f:
            pickle.dump(id_aminer, f)
    Nt = len(id_aminer)
    # --------------------- obtain the underlying identities -----------------------
    lying_t_full_path = os.path.join(raw_oag_fold, "lying_t_full.pt")
    if os.path.exists(lying_t_full_path):
        print("loading the underlying identities")
        with open(lying_t_full_path, "rb") as f:
            lying_t_full = pickle.load(f)
    else:
        print("obtaining the underlying identities")
        lying_t_full = []
        for aid in id_aminer:
            lying_t_full.append(aid)
        with open(lying_t_full_path, "wb") as f:
            pickle.dump(lying_t_full, f)

    # --------------------------- obtain the type of each node ----------------------
    node_types_t_full_path = os.path.join(raw_oag_fold, "node_types_t_full.npy")
    if os.path.exists(node_types_t_full_path):
        print("loading node_types_t")
        node_types_t_full = np.load(node_types_t_full_path)
    else:
        print("obtaining the node_types_t_full")
        node_types_t_full = np.zeros(Nt, dtype=np.int)
        for path in aminer_author_paths:
            print("path={}".format(path))
            for line in open(path, "r"):
                tmp_d = ast.literal_eval(line)
                index = id_aminer.bisect_left(tmp_d["id"])
                node_types_t_full[index] = author_type
        for path in aminer_paper_paths:
            print("path={}".format(path))
            for line in open(path, "r"):
                tmp_d = ast.literal_eval(line)
                index = id_aminer.bisect_left(tmp_d["id"])
                node_types_t_full[index] = paper_type
        for path in aminer_venue_paths:
            print("path={}".format(path))
            for line in open(path, "r"):
                tmp_d = ast.literal_eval(line)
                index = id_aminer.bisect_left(tmp_d["id"])
                node_types_t_full[index] = venue_type
        np.save(node_types_t_full_path, node_types_t_full)

    # --------------------------- obtain values and adjacencies ---------------------
    values_t_full_path = os.path.join(raw_oag_fold, "values_t_full.npy")
    adj_t_full_path = os.path.join(raw_oag_fold, "adj_t_full.npz")
    if os.path.exists(values_t_full_path) and os.path.exists(adj_t_full_path):
        print("loading values and adjacencies")
        values_t_full = np.load(values_t_full_path)
        adj_t_full = load_npz(adj_t_full_path)
    else:
        print("obtaining values and adjacencies")
        values_t_full = np.zeros([Nt, 32])
        for path in aminer_author_paths:
            print("path={}".format(path))
            for line in open(path, "r"):
                tmp_d = ast.literal_eval(line)
                index = id_aminer.bisect_left(tmp_d["id"])
                try:
                    values_t_full[index] = count(tmp_d["name"])
                except KeyError:
                    pass
        for path in aminer_venue_paths:
            print("path={}".format(path))
            for line in open(path, "r"):
                tmp_d = ast.literal_eval(line)
                index = id_aminer.bisect_left(tmp_d["id"])
                try:
                    values_t_full[index] = count(tmp_d["DisplayName"])
                except KeyError:
                    pass
        weights, rows, cols = [], [], []
        for path in aminer_paper_paths:
            print("path={}".format(path))
            for line in open(path, "r"):
                tmp_d = ast.literal_eval(line)
                i = id_aminer.bisect_left(tmp_d["id"])
                try:
                    values_t_full[i] = count(tmp_d["title"])
                except KeyError:
                    pass
                # edges between paper nodes and author nodes
                for author_id in tmp_d["authors"]:
                    j = id_aminer.bisect_left(author_id)
                    if j < Nt:
                        weights.append(1)
                        weights.append(1)
                        rows.append(i)
                        rows.append(j)
                        cols.append(j)
                        cols.append(i)
                try:
                    venue_id = tmp_d["venue"]
                    j = id_aminer.bisect_left(venue_id)
                    if j < Nt:
                        weights.append(1)
                        weights.append(1)
                        rows.append(i)
                        rows.append(j)
                        cols.append(j)
                        cols.append(i)
                except KeyError:
                    pass
        adj_t_full = csr_matrix((weights, (rows, cols)), shape=(Nt, Nt))
        np.save(values_t_full_path, values_t_full)
        save_npz(adj_t_full_path, adj_t_full)
    # ============================= constructing the graph pair ===============================
    # ---------------------------- constructing the reverse links -----------------------------
    r_links = SortedDict()
    for mid in links.keys():
        aid = links[mid]
        r_links[aid] = mid
    deg_s = np.ravel(adj_s_full.sum(axis=1))
    print("deg_s={}".format(deg_s))

    while True:
        # ini_s = np.random.choice(Ns)
        ini_s = np.random.choice(Ns, p=deg_s / np.sum(deg_s))
        print("ini_s={}".format(ini_s))
        # if node_types_s_full[ini_s] != venue_type:
        #     continue
        if not overlapped(index=ini_s, node_ids=id_mag, links=links):
            continue
        id_s = id_mag[ini_s]
        id_t = links[id_s]
        ini_t = id_aminer.bisect_left(id_t)
        if ini_t < Nt:
            break

    # print("=========== obtaining the source graph ================")
    # indices_s = reasonable_size(adj_s_full, node_types=node_types_s_full, node_ids=id_mag, links=links,
    #                             min_size=min_size, max_size=max_size, ini_node=ini_s)
    # print("=========== obtaining the target graph ================")
    # indices_t = reasonable_size(adj_t_full, node_types=node_types_t_full, node_ids=id_aminer, links=r_links,
    #                             min_size=min_size, max_size=max_size, ini_node=ini_t)
    ns = 2 * min_size
    nt = ns
    indices_s, indices_t = [ini_s], [ini_t]
    used_s, used_t = [], []
    print("=========== obtaining the source graph ================")
    while True:
        for i in indices_s:
            if i not in used_s:
                nei = get_nei(w=adj_s_full, i=i)
                for j in nei:
                    if j not in indices_s:
                        if overlapped(index=j, node_ids=id_mag, links=links):
                            indices_s.append(j)
                            j_star = id_aminer.bisect_left(links[id_mag[j]])
                            indices_t.append(j_star)
                            assert j_star < Nt
                        else:
                            if np.random.uniform(0, 1) > add_percent:  # include more overlapped nodes
                                indices_s.append(j)
                        if len(indices_s) >= ns:
                            break
            used_s.append(i)
            if len(indices_s) >= ns:
                break
        print("len(indices_s)={}".format(len(indices_s)))
        if len(indices_s) >= ns:
            display(indices=indices_s, node_types=node_types_s_full)
            break
    print("=========== obtaining the target graph ================")
    while True:
        for i1 in indices_t:
            if i1 not in used_t:
                nei = get_nei(w=adj_t_full, i=i1)
                for j1 in nei:
                    if j1 not in indices_t:
                        indices_t.append(j1)
                        if len(indices_s) >= nt:
                            break
            used_t.append(i1)
            if len(indices_t) >= nt:
                break
        if len(indices_t) >= nt:
            display(indices=indices_t, node_types=node_types_t_full)
            break
    ws = adj_s_full[indices_s][:, indices_s].toarray()
    wt = adj_t_full[indices_t][:, indices_t].toarray()
    lying_s = [lying_s_full[i] for i in indices_s]
    lying_t = [lying_t_full[i] for i in indices_t]
    graph_s = graph(w=ws, lying=lying_s)
    graph_t = graph(w=wt, lying=lying_t)
    graph_s.set_node_types(node_types=node_types_s_full[indices_s])
    graph_t.set_node_types(node_types=node_types_t_full[indices_t])
    graph_s.set_values(values=values_s_full[indices_s])
    graph_t.set_values(values=values_t_full[indices_t])
    n_overlap = 0
    for a in lying_s:
        for b in lying_t:
            if a == b:
                n_overlap += 1
                break
    graph_st = graph_pair_rn(graph_s=graph_s, graph_t=graph_t, n_overlap=n_overlap, anch_ratio=0.01)
    print(graph_st.result_eval(graph_st.gt))
    mu_s = np.sum(graph_st.graph_s.w, axis=1)
    mu_t = np.sum(graph_st.graph_t.w, axis=1)
    print(np.sum(mu_s == 0))
    print(np.sum(mu_t == 0))
    data_path = os.path.join("data", "oag", "{}_{}_{}_{}_sd{}.p".format(n_overlap, len(lying_s), len(lying_t), 1, seed))
    with open(data_path, "wb") as f:
        pickle.dump(graph_st, f)
    return graph_st, data_path


if __name__ == '__main__':
    # author_nodes_pre()
    # paper_nodes_pre()
    main_api_type()
