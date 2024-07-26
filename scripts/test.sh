#!/bin/bash
data_name="tcga_ov_os"
cancer_type="TCGA_OV"
omics="rna"
output_file="./scripts/error_output.txt"

> "$output_file"

run_command() {
  local cmd="$1"
  echo "Running: $cmd"
  eval "$cmd" || (echo "Command failed, capturing output..." && eval "$cmd >> $output_file 2>&1")
}


# Omic + WSI: bilinear fusion
run_command "CUDA_VISIBLE_DEVICES=0 python run_mmsurv.py --results_dir results/testing/${data_name} --data_name ${data_name} --feats_dir /media/nfs/SURV/${cancer_type}/Feats1024/UNI/ --model_type porpoise --selected_features --omics ${omics} --max_epochs 1 --k 1"
run_command "CUDA_VISIBLE_DEVICES=0 python run_mmsurv.py --results_dir results/testing/${data_name} --data_name ${data_name} --feats_dir /media/nfs/SURV/${cancer_type}/Feats1024/UNI/ --model_type amil --selected_features --omics ${omics} --max_epochs 1 --k 1"
run_command "CUDA_VISIBLE_DEVICES=0 python run_mmsurv.py --results_dir results/testing/${data_name} --data_name ${data_name} --feats_dir /media/nfs/SURV/${cancer_type}/Feats1024/UNI/ --model_type deepset --selected_features --omics ${omics} --max_epochs 1 --k 1"
run_command "CUDA_VISIBLE_DEVICES=0 python run_mmsurv.py --results_dir results/testing/${data_name} --data_name ${data_name} --feats_dir /media/nfs/SURV/${cancer_type}/Feats1024/UNI/ --model_type deepattnmisl --selected_features --omics ${omics} --max_epochs 1 --k 1"

# Omic + WSI: coattention
run_command "CUDA_VISIBLE_DEVICES=0 python run_mmsurv.py --results_dir results/testing/${data_name} --data_name ${data_name} --feats_dir /media/nfs/SURV/${cancer_type}/Feats1024/UNI/ --model_type mcat --apply_sig --selected_features --omics ${omics} --max_epochs 1 --k 1"
run_command "CUDA_VISIBLE_DEVICES=0 python run_mmsurv.py --results_dir results/testing/${data_name} --data_name ${data_name} --feats_dir /media/nfs/SURV/${cancer_type}/Feats1024/UNI/ --model_type motcat --apply_sig --selected_features --omics ${omics} --max_epochs 1 --k 1"
