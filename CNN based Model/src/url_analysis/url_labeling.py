import pandas as pd

def label_urls():
    """Label extracted URLs based on PhishTank dataset."""
    try:
        print("🔹 Loading QR-extracted URLs dataset...")
        qr_urls_df = pd.read_csv("./data/qr_extracted_urls.csv")
        print(f" QR URLs Loaded: {qr_urls_df.shape}")

        print("🔹 Loading PhishTank dataset...")
        phishtank_df = pd.read_csv("./data/phishtank_data.csv")
        print(f" PhishTank Data Loaded: {phishtank_df.shape}")

        # Check if required columns exist
        if "URL" not in qr_urls_df.columns:
            raise ValueError("Column 'URL' missing in qr_extracted_urls.csv")
        if "url" not in phishtank_df.columns:
            raise ValueError("Column 'url' missing in phishtank_data.csv")

        # Convert URLs to lowercase for case-insensitive comparison
        qr_urls_df["URL"] = qr_urls_df["URL"].astype(str).str.lower()
        phishtank_df["url"] = phishtank_df["url"].astype(str).str.lower()

        # Create a set of known phishing URLs
        phishing_urls = set(phishtank_df["url"])

        # Assign labels: 1 if phishing, else 0
        qr_urls_df["label"] = qr_urls_df["URL"].apply(lambda url: 1 if url in phishing_urls else 0)

        # Save labeled dataset
        labeled_file_path = "./data/labeled_url_data.csv"
        qr_urls_df.to_csv(labeled_file_path, index=False)

        print(f" Labeled URL dataset saved to {labeled_file_path}")

    except Exception as e:
        print(f" Error in labeling URLs: {e}")

if __name__ == "__main__":
    label_urls()
