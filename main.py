import argparse
from module1_preprocessing import load_and_preprocess
from module2_features import build_features
from module3_classifier import train_and_evaluate

def main(data_dir="data", method="hog", use_pca=False):
    #preprocessing
    X_train, y_train, X_val, y_val, X_test, test_ids = load_and_preprocess(data_dir)

    #feature extraction
    X_train_feat, X_val_feat, X_test_feat, _ = build_features(
        X_train, y_train, X_val, X_test, method=method, use_pca=use_pca, visualize=True
    )

    #train and evaluation
    _, metrics = train_and_evaluate(
        X_train_feat, y_train, X_val_feat, y_val, X_test_feat, test_ids, run_knn_baseline=True
    )

    print(f"\nAccuracy: {metrics['accuracy']:.4f}\nOutputs saved to outputs/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IIVP2026 digit recognition pipeline")
    parser.add_argument("--data-dir", default="data",  help="Path to data directory")
    parser.add_argument("--method",   default="hog",   choices=["hog", "histogram", "combined"])
    parser.add_argument("--pca",      action="store_true", help="Apply PCA after feature extraction")
    args = parser.parse_args()

    main(data_dir=args.data_dir, method=args.method, use_pca=args.pca)
