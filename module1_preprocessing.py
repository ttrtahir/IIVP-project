import os, glob, cv2
import numpy as np
from skimage import data_dir
from skimage import data_dir
from sklearn.model_selection import train_test_split

NUM_CLASSES = 10
IMG_SIZE = 32
CACHE_DIR = "cache"

def load_raw_images(data_dir="data"):
    train_dir = os.path.join(data_dir, "train", "train")
    test_dir = os.path.join(data_dir, "test", "test")
    
    train_images, train_labels = [], []
    for label in range(NUM_CLASSES):
        cls_path = os.path.join(train_dir, str(label))
        files = sorted(glob.glob(os.path.join(cls_path, "*.png")))
        for f in files:
            img = cv2.imread(f)
            if img is not None:
                train_images.append(img)
                train_labels.append(label)

    test_ids = [os.path.splitext(os.path.basename(p))[0] 
                for p in sorted(glob.glob(os.path.join(test_dir, "*.png")))]
    
    test_images = [cv2.imread(os.path.join(test_dir, f"{i}.png")) for i in test_ids]
            
    return train_images, np.array(train_labels), test_images, test_ids

def preprocess_image(img):
    # convert to gray 
    if img.ndim == 3:
        if img.shape[2] == 4: img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0

    #denoise bilateral
    img_u8 = (img * 255).astype(np.uint8)
    denoised = cv2.bilateralFilter(img_u8, 9, 75, 75)

    #threshold
    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    return cleaned.astype(np.float32) / 255.0

def load_and_preprocess(data_dir="data"):

    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_files = ["X_train.npy", "y_train.npy", "X_val.npy", "y_val.npy", "X_test.npy", "test_ids.npy"]
    
    if all(os.path.exists(os.path.join(CACHE_DIR, f)) for f in cache_files):
        return [np.load(os.path.join(CACHE_DIR, f), allow_pickle=True) for f in cache_files]

    raw_tr, labels, raw_te, ids = load_raw_images(data_dir)

    X_all = np.array([preprocess_image(im) for im in raw_tr])
    X_test = np.array([preprocess_image(im) for im in raw_te])

    X_train, X_val, y_train, y_val = train_test_split(
        X_all, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Save 
    outputs = [X_train, y_train, X_val, y_val, X_test, np.array(ids)]
    for name, data in zip(cache_files, outputs):
        np.save(os.path.join(CACHE_DIR, name), data)

    return outputs