import os
import pandas as pd
import numpy as np
import torch
import h5py

# Generate dummy data for patients and slides
num_patients = 30
min_shape = 200
max_shape = 5000

slide_ids = []
case_ids = []
shapes = []

dummy_split = pd.DataFrame({
    "train": list(np.random.randint(0, num_patients, int(num_patients*0.8))),
    "val": list(np.random.randint(0, num_patients, int(num_patients*0.2))) + [pd.NA] * int(num_patients*0.6),
    "test": list(np.random.randint(0, num_patients, int(num_patients*0.2))) + [pd.NA] * int(num_patients*0.6)
})
os.makedirs("./datasets_csv/", exist_ok=True)
os.makedirs("./splits/dummy/", exist_ok=True)
dummy_split.to_csv("./splits/dummy/splits_0.csv", index=False)

dummy_df = pd.DataFrame({
    "survival_months": np.random.randint(1, 200, size=num_patients),
    "event": np.random.randint(0, 2, size=num_patients),
    "case_id": np.arange(num_patients),
})
dummy_df["censorship"] = 1 - dummy_df["event"]

dummy_rna_df = pd.DataFrame({f"dummy{i}_rna": np.random.randn(num_patients,) for i in range(5)})
dummy_rna_df["case_id"] = np.arange(num_patients)
dummy_df = pd.merge(dummy_df, dummy_rna_df, on="case_id")

dummy_dna_df = pd.DataFrame({f"dummy{i}_dna": np.random.randn(num_patients,) for i in range(10)})
dummy_dna_df["case_id"] = np.arange(num_patients)
dummy_df = pd.merge(dummy_df, dummy_dna_df, on="case_id")

dummy_cnv_df = pd.DataFrame({f"dummy{i}_cnv": np.random.randn(num_patients,) for i in range(15)})
dummy_cnv_df["case_id"] = np.arange(num_patients)
dummy_df = pd.merge(dummy_df, dummy_cnv_df, on="case_id")

for patient_id in range(num_patients):
    num_slides = np.random.randint(2, 4)
    for slide_num in range(num_slides):
        slide_id = f"slide_id{patient_id + 1}_{slide_num + 1}"
        slide_ids.append(slide_id)
        case_ids.append(patient_id)
        shapes.append(np.random.randint(min_shape, max_shape))

os.makedirs("./dummy_data/feats_dir/", exist_ok=True)
os.makedirs("./dummy_data/coords_dir/", exist_ok=True)

for slide_id, shape in zip(slide_ids, shapes):
    feats = torch.randn(shape, 768)  
    torch.save(feats, f"./dummy_data/feats_dir/{slide_id}.pt")
    
    coords = np.random.rand(shape, 2)  
    with h5py.File(f"./dummy_data/coords_dir/{slide_id}.h5", 'w') as h5f:
        h5f.create_dataset('coords', data=coords)

dummy_slide_df = pd.DataFrame({
    "slide_id": slide_ids,
    "case_id": case_ids
})

dummy_df = pd.merge(dummy_df, dummy_slide_df, on="case_id")
dummy_df.to_csv("./datasets_csv/dummy_selected.csv", index=False)

print("Dummy data generated and saved.")
