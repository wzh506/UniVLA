from huggingface_hub import snapshot_download

# 只下载某个子文件夹
snapshot_download(
    repo_id="TRI-ML/prismatic-vlms",
    allow_patterns="prism-dinosiglip-224px+7b/*",
    local_dir="/home/lucian.wang/github/UniVLA/ckpt/prismatic-vlms"
)

# # 只下载某个子文件夹
# snapshot_download(
#     repo_id="TRI-ML/prismatic-vlms",
#     allow_patterns="prism-dinosiglip-224px+7b/*",
#     local_dir="/home/lucian.wang/github/UniVLA/ckpt"
# )