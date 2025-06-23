import os
from huggingface_hub import Repository

def deploy():
    token = os.getenv("HF_API_KEY")  # récupère le token depuis les secrets GitHub Actions
    repo_id = "IssaBachir/test2"

    repo = Repository(local_dir="models", clone_from=repo_id, use_auth_token=token)
    repo.push_to_hub(commit_message="Déploiement modèle")

if __name__ == "__main__":
    deploy()
