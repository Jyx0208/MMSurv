import argparse
import numpy as np
import pickle
import os
import h5py
import pandas as pd
from sklearn.cluster import KMeans


args = argparse.ArgumentParser()
args.add_argument("data_name")
args.add_argument("--dataset_dir", type=str, default="./datasets_csv/")
args.add_argument("--patch_dir", type=str, default=None)
args.add_argument("--n_clusters", type=int, default=10)
args.add_argument("--seed", type=int, default=42)

args = args.parse_args()

df = pd.read_csv(os.path.join(args.dataset_dir, args.data_name+"_selected.csv"))
cancer_type = args.data_name.rsplit("_", 1)[0].upper()
patch_dir = f"/media/nfs/SURV/{cancer_type}/SP1024/patches/" if not args.patch_dir else args.patch_dir
cluster_ids = {}
for slide in df["slide_id"].values:
	with h5py.File(os.path.join(patch_dir, slide+".h5"), 'r') as f:
		coords = np.array(f['coords'])
	kmeans = KMeans(n_clusters=args.n_clusters, random_state=args.seed).fit(coords)
	cluster_ids[slide] = kmeans.labels_

with open(os.path.join(args.dataset_dir, args.data_name+"_cluster_ids.pkl"), "wb") as f:
    pickle.dump(cluster_ids, f)