from huggingface_hub import HfApi, HfFolder, Repository
import os
import shutil

model_path = "models/model"
repo_name = "test2-model-deploy"

hf_token = os.environ.get("HF_TOKEN")
assert hf_token is not None, "Clé Hugging Face manquante"

# Crée le repo si besoin
api = HfApi()
if not any(r.repo_id == f"IssaBachir/{repo_name}" for r in api.list_repos_objs(token=hf_token)):
    api.create_repo(name=repo_name, token=hf_token)

# Pousser le modèle
repo_url = f"https://huggingface.co/IssaBachir/{repo_name}"
repo_dir = "./models/repo"
if os.path.exists(repo_dir):
    shutil.rmtree(repo_dir)

repo = Repository(local_dir=repo_dir, clone_from=repo_url, use_auth_token=hf_token)
repo.git_pull()

shutil.copytree(model_path, repo_dir, dirs_exist_ok=True)
repo.push_to_hub()
