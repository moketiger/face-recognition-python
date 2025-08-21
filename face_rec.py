import os
import torch
import numpy as np
import pickle
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

# Initialize detector (MTCNN) and recognizer (FaceNet)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# --- Function: get embeddings for faces in an image ---
def get_face_embeddings(img):
    faces = mtcnn(img)
    if faces is None:
        return []
    with torch.no_grad():
        return resnet(faces.to(device)).cpu().numpy()

# --- Function: register known faces from folder ---
def register_faces(known_faces_dir="known_faces"):
    valid_exts = (".jpg", ".jpeg", ".png")
    embeddings = {}
    for person_name in os.listdir(known_faces_dir):
        person_folder = os.path.join(known_faces_dir, person_name)
        if not os.path.isdir(person_folder):
            continue

        person_embeddings = []
        for img_name in os.listdir(person_folder):
            img_path = os.path.join(person_folder, img_name)

            # skip if it's not a file or not an image
            if not os.path.isfile(img_path) or not img_name.lower().endswith(valid_exts):
                continue
            
            try:
                img = Image.open(img_path).convert("RGB")
                emb_list = get_face_embeddings(img)
                if len(emb_list) > 0:
                    person_embeddings.append(emb_list[0])
            except Exception as e:
                print(f"⚠️ Skipping {img_path}: {e}")

        if len(person_embeddings) > 0:
            # Average embeddings for better stability
            embeddings[person_name] = np.mean(person_embeddings, axis=0)

    return embeddings

# --- Function: recognize faces in test image ---
def recognize_faces(known_embeddings, test_img_folder_path, threshold=0.5):

    for test_img in os.scandir(test_img_folder_path):
        test_img_path = test_img.path
        img = Image.open(test_img_path).convert("RGB")
        test_emb_list = get_face_embeddings(img)

        if len(test_emb_list) == 0:
            os.replace(test_img_path , r"C:\Users\lenovo\Pictures\images_without_family")
            return

        Known_faces_amount: int = 0
        for i, test_emb in enumerate(test_emb_list):
            best_score = -1

            for name, emb in known_embeddings.items():
                sim = np.dot(emb, test_emb) / (np.linalg.norm(emb) * np.linalg.norm(test_emb))
                if sim > best_score:
                    best_score = sim

            if best_score > threshold:
                Known_faces_amount += 1
            else:
                print(f"Face {i+1}: Unknown (score={best_score:.2f})")
        
        if Known_faces_amount > 0:
            os.replace(
                test_img_path,
                os.path.join(r"C:\Users\lenovo\Pictures\images_with_family", os.path.basename(test_img_path))
            )
                    
        else:
            os.replace(
                test_img_path,
                os.path.join(r"C:\Users\lenovo\Pictures\images_without_family", os.path.basename(test_img_path))
            )

# --- Main ---
if __name__ == "__main__":
    EMB_FILE = r"C:\gilad_codeing\python_projects\face_recognizer\embeddings.pkl"

    if os.path.exists(EMB_FILE):
        print("🔹 Loading saved embeddings...")
        with open(EMB_FILE, "rb") as f:
            embeddings = pickle.load(f)
    else:
        print("🔹 Registering new faces from folder...")
        embeddings = register_faces(r"C:\gilad_codeing\python_projects\face_recognizer")
        with open(EMB_FILE, "wb") as f:
            pickle.dump(embeddings, f)

    # Test on an image
    recognize_faces(embeddings, r"C:\gilad_codeing\python_projects\face_recognizer\test_images")

print("everything finsh successfully!")