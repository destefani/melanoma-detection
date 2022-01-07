import os
import requests
from tqdm import tqdm

# ISIC 2020 jpeg data
isic_2020_images = (
    "https://isic-challenge-data.s3.amazonaws.com/2020/ISIC_2020_Training_JPEG.zip"
)
isic_2020_metadata_v1 = "https://isic-challenge-data.s3.amazonaws.com/2020/ISIC_2020_Training_GroundTruth.csv"
isic_2020_metadata_v2 = "https://isic-challenge-data.s3.amazonaws.com/2020/ISIC_2020_Training_GroundTruth_v2.csv"
isic_2020_duplicates = "https://isic-challenge-data.s3.amazonaws.com/2020/ISIC_2020_Training_Duplicates.csv"

isic_2020_dataset = [
    isic_2020_images,
    isic_2020_metadata_v1,
    isic_2020_metadata_v2,
    isic_2020_duplicates,
]


def download_file(url):
    """Downloads a file from a url"""
    local_filename = f"data/{url.split('/')[-1]}"
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size_in_bytes = int(r.headers.get("content-length", 0))
        block_size = 1024
        progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
        with open(local_filename, "wb") as f:
            for data in r.iter_content(block_size):
                progress_bar.update(len(data))
                f.write(data)
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            raise ("Download error")
        else:
            print(f"Downloaded {local_filename}")
    return local_filename


if __name__ == "__main__":
    if not os.path.exists("data"):
        os.mkdir("data")
    # Check if data is already downloaded
    for file in isic_2020_dataset:
        if not os.path.exists(f"data/{file.split('/')[-1]}"):
            print(f"Downloading {file}")
            download_file(file)
