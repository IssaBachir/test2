from huggingface_hub import Repository
import os

def deploy():
    token = os.getenv("HF_API_KEY")
    repo_id = "IssaBachir/test2"  # Remplace par ton vrai repo HF

    # ✅ Nouveau dossier de clone vide
    repo_dir = "hf_repo"

    # Clone dans un dossier vide (à la place de 'models')
    repo = Repository(local_dir=repo_dir, clone_from=repo_id, use_auth_token=token)

    # Déplace le modèle entraîné dans ce dossier
    import shutil
    shutil.copy("models/model_final.pth", f"{repo_dir}/model_final.pth")

    # Commit & push
    repo.push_to_hub(commit_message="Ajout du modèle final")

if __name__ == "__main__":
    deploy()
