from huggingface_hub import Repository
import os
import shutil

def deploy():
    token = os.getenv("HUGGINGFACE_TOKEN")
    repo_id = "issabachir6/test2"  # ✅ corriger ici

    repo_dir = "hf_repo"
    if os.path.exists(repo_dir):
        shutil.rmtree(repo_dir)

    repo = Repository(local_dir=repo_dir, clone_from=repo_id, use_auth_token=token)

    # Copier le modèle dans le dossier du repo
    shutil.copy("models/model_final.pth", f"{repo_dir}/model_final.pth")

    # Commit & push
    repo.push_to_hub(commit_message="Déploiement automatique du modèle")

if __name__ == "__main__":
    deploy()
