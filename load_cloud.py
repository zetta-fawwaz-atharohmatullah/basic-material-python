from google.cloud import storage
from dotenv import load_dotenv
import os
from load_pdf import load_pdf, read_pdf_content

load_dotenv()

## get bucket and file name
def get_bucket_name():
    """_summary_

    Returns:
        _type_: _description_
    """
    bucket_name = os.getenv("bucket_name")  
    file_name = os.getenv("blob_name")
    return bucket_name, file_name

# get env service acc key
def get_service_acc():
    """_summary_

    Returns:
        _type_: _description_
    """
    service_acc_key = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not service_acc_key:
        raise Exception("GOOGLE_APPLICATION_CREDENTIALS not seen in env")
    return service_acc_key

# storage bucket initialize
def initialize_storage_client():
    """_summary_

    Returns:
        _type_: _description_
    """
    get_service_acc()
    return storage.Client()

# download pdf file from bucket
def download_pdf_from_bucket(storage_client, bucket_name, blob_name):
    """_summary_

    Args:
        storage_client (_type_): _description_
        bucket_name (_type_): _description_
        blob_name (_type_): _description_

    Returns:
        _type_: _description_
    """
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    local_pdf_path = "temp_downloaded_file.pdf"  
    blob.download_to_filename(local_pdf_path)
    print(f"File downloaded to {local_pdf_path}")
    return local_pdf_path

# main to execute all the functions above and print out the text
def main():   
    """
    _summary_
    main function to execute all functions
    """
    bucket_name, blob_name = get_bucket_name()
    storage_client = initialize_storage_client()
    pdf_path = download_pdf_from_bucket(storage_client, bucket_name, blob_name)
    content = load_pdf(pdf_path)
    text = read_pdf_content(content)
    print(text)
    
if __name__ == "__main__":
    main()