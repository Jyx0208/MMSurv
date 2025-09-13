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

# 创建正确的数据分割
np.random.seed(42)  # 为了可重复性

# 创建train/val/test分割
train_size = int(num_patients * 0.7)
val_size = int(num_patients * 0.15) 
test_size = num_patients - train_size - val_size

# 随机分割患者ID
all_patients = list(range(num_patients))
np.random.shuffle(all_patients)

train_patients = all_patients[:train_size]
val_patients = all_patients[train_size:train_size + val_size]
test_patients = all_patients[train_size + val_size:]

# 创建分割DataFrame，每行一个患者
max_split_size = max(len(train_patients), len(val_patients), len(test_patients))

dummy_split = pd.DataFrame({
    "train": train_patients + [pd.NA] * (max_split_size - len(train_patients)),
    "val": val_patients + [pd.NA] * (max_split_size - len(val_patients)),
    "test": test_patients + [pd.NA] * (max_split_size - len(test_patients))
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

# 添加病理组学数据
dummy_path_df = pd.DataFrame({f"dummy{i}_pat": np.random.randn(num_patients,) for i in range(20)})  # 20维病理特征
dummy_path_df["case_id"] = np.arange(num_patients)
dummy_df = pd.merge(dummy_df, dummy_path_df, on="case_id")

# 确保没有重复的列
dummy_df = dummy_df.loc[:, ~dummy_df.columns.duplicated()]

# 保存单独的病理数据文件
dummy_path_df.to_csv("./datasets_csv/dummy_pat.csv.zip", compression="zip", index=False)

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
