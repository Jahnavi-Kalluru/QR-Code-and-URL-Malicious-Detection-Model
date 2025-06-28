import cv2
import pyzbar.pyzbar as pyzbar
import pandas as pd
import os

def extract_qr_data(image_folder, output_file):
    """Extract URLs from QR codes in images inside a folder."""
    urls = []
    for filename in os.listdir(image_folder):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            img = cv2.imread(os.path.join(image_folder, filename))
            decoded_objects = pyzbar.decode(img)
            for obj in decoded_objects:
                urls.append(obj.data.decode("utf-8"))

    df = pd.DataFrame({"URL": urls})
    df.to_csv(output_file, index=False)
    return f"Extracted {len(urls)} URLs and saved to {output_file}"

if __name__ == "__main__":
    print(extract_qr_data("./Mixed_qr_images_1000", "./data/qr_extracted_urls.csv"))
