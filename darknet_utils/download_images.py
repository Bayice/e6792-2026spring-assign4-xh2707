"""
Google Images Download.

E6792 Spring 2026

YOU DO NOT NEED TO MODIFY THIS FUNCTION TO COMPLETE THE ASSIGNMENT
"""
import os
import requests


DEPENDANCIES = ['tqdm',
                'BeautifulSoup4']

def install_dependancies():
    """
    Install the dependancies listed above with pip.
    """
    for dependancy in DEPENDANCIES:
        os.system('pip3 install {}'.format(dependancy))
        print("{} installed.".format(dependancy))

try:
    from bs4 import BeautifulSoup
except:
    install_dependancies()
    from bs4 import BeautifulSoup
    
DOWNLOADS_PATH = './downloads/'

def download_images(query, num_images):
    """
    Download the first N google images into the "downloads" folder
    of the root directory.
    
    params:
        query (string) : Google image query
        num_images (int) : Number of images to retrieve
    """

    query = query.replace(' ', '+')

    # URL for Google Images
    url = f"https://www.google.com/search?q={query}&tbm=isch"

    # Make a request to the website
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find all image tags
    img_tags = soup.find_all("img")

    # Create a directory to save images
    folder_name = query.replace('+', '_')
    save_path = os.path.join(DOWNLOADS_PATH, folder_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Download images
    count = 0
    for img in img_tags:
        # Stop when we have enough images
        if count >= num_images:
            break

        # Get image URL
        img_url = img['src']
        try:
            # Send a request to the image URL
            img_response = requests.get(img_url)

            # Save the image
            with open(f"{save_path}/{count}.jpg", 'wb') as f:
                f.write(img_response.content)
            count += 1

        except:
            # Skip if there's any issue with one image
            pass