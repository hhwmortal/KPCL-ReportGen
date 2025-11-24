import subprocess

# 定义训练的命令行参数
command = [
    "python", "main_train.py",
    "--image_dir", "data/PED_xray/images/",
    "--ann_path", "data/PED_xray/annotation.json",
    "--dataset_name", "PED_xray",
    "--max_seq_length", "100",
    "--threshold", "3",
    "--epochs", "100",
    "--batch_size", "16",
    "--step_size", "10",
    "--gamma", "0.8",
    "--num_layers", "3",
    "--seed", "7580",
    "--beam_size", "3",
    "--save_dir", "results/PED_xray/",
]

# 通过 subprocess 运行命令
subprocess.run(command)