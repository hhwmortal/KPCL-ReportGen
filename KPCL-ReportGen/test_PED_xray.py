import subprocess

# 定义训练的命令行参数
command = [
    "python", "main_test.py",
    "--image_dir", "data/PED_xray/images/",
    "--ann_path", "data/PED_xray/annotation.json",
    "--dataset_name", "PED_xray",
    "--max_seq_length", "100",
    "--threshold", "3",
    "--epochs", "100",
    '--save_dir', 'results/PED_xray/',
    "--batch_size", "16",
    "--step_size", "10",
    "--gamma", "0.8",
    "--save_dir", "results/PED_xray/",
    '--load', 'results/PED_xray/model_best.pth'
]

# 通过 subprocess 运行命令
subprocess.run(command)