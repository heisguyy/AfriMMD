from huggingface_hub import HfApi

# Initialize Hugging Face API
api = HfApi()

# Create repository (if it doesn't exist)
repo_id = "AfriMM/SiglipNllb"
api.create_repo(repo_id, exist_ok=True)

# Upload the weights file
api.upload_file(
    path_or_fileobj="outputs/checkpoint_epoch_15.pth",
    path_in_repo="model.pth",
    repo_id=repo_id,
    commit_message="Upload Siglip pytorch weights"
)