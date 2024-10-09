import requests
from bs4 import BeautifulSoup
import os
import re

# Define the URL of the Kubernetes documentation to scrape
BASE_URL = "https://kubernetes.io"
DOC_URL = f"{BASE_URL}/docs/reference/generated/kubernetes-api/v1.20/"

# Create a folder to save the scraped YAML files
OUTPUT_FOLDER = "kubernetes_yaml_docs"
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# Function to clean and format file names
def clean_filename(filename):
    filename = re.sub(r'[\\/*?:"<>|]', "", filename)
    return filename.replace(" ", "_")

# Function to scrape the page
def scrape_yaml_examples(url):
    try:
        # Send HTTP GET request
        response = requests.get(url)
        response.raise_for_status()
        
        # Parse HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find YAML examples in the page (YAML is often wrapped in <pre><code> tags)
        yaml_examples = soup.find_all("pre", {"class": "language-yaml"})
        
        if yaml_examples:
            print(f"Found {len(yaml_examples)} YAML examples on {url}")
            
            # Extract YAML examples and save them to files
            for idx, example in enumerate(yaml_examples):
                yaml_content = example.get_text()
                doc_title = soup.find("h1").get_text()
                filename = f"{clean_filename(doc_title)}_example_{idx+1}.yaml"
                filepath = os.path.join(OUTPUT_FOLDER, filename)
                
                with open(filepath, "w") as file:
                    file.write(yaml_content)
                print(f"Saved {filename}")
                
        else:
            print(f"No YAML examples found on {url}")
    
    except requests.RequestException as e:
        print(f"Failed to retrieve page {url}: {e}")

# Function to scrape all relevant pages in the Kubernetes API reference
def scrape_kubernetes_docs():
    try:
        # Get the list of all API resources pages
        response = requests.get(DOC_URL)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find all links to API resource pages
        api_links = soup.find_all("a", href=True)
        for link in api_links:
            href = link['href']
            if href.startswith("/docs/reference/generated/kubernetes-api/v1.20/"):
                scrape_url = f"{BASE_URL}{href}"
                print(f"Scraping {scrape_url} ...")
                scrape_yaml_examples(scrape_url)
    
    except requests.RequestException as e:
        print(f"Failed to retrieve Kubernetes docs main page: {e}")

if __name__ == "__main__":
    scrape_kubernetes_docs()
