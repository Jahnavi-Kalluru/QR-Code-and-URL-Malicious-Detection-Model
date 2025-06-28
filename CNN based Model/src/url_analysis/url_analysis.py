import requests
import pandas as pd
import re
from urllib.parse import urlparse

# API Endpoints
PHISHTANK_API_URL = "http://data.phishtank.com/data/online-valid.csv"
URLHAUS_API_URL = "https://urlhaus.abuse.ch/downloads/csv_recent/"

# Suspicious keywords found in analyzed URLs
SUSPICIOUS_KEYWORDS = [
    "suspicious", "account", "login", "secure", "support", "auth", "free",
    "claim", "prize", "win", "bank", "wallet", "pay", "password", "signin",
    "cloud", "hookups", "credit", "card", "apple", "facebook", "telegram", "alert"
]

# Lookalike domain patterns
LOOKALIKE_PATTERNS = [
    r"g[0o]{2}gle", r"rnicrosoft", r"facebo[0o]k", r"paypa[1l]", r"amaz[0o]n", r"y0utube"
]

# Risky TLDs (Top-Level Domains)
RISKY_TLDS = ["cc", "tk", "ml", "pw", "ga", "cf", "gq"]

def fetch_phishtank_data():
    """Fetch latest phishing URLs from PhishTank."""
    response = requests.get(PHISHTANK_API_URL)
    with open("./data/phishtank_data.csv", "wb") as file:
        file.write(response.content)
    return pd.read_csv("./data/phishtank_data.csv")

def fetch_urlhaus_data():
    """Fetch latest malware URLs from URLHaus."""
    response = requests.get(URLHAUS_API_URL)
    with open("./data/urlhaus_data.csv", "wb") as file:
        file.write(response.content)
    return pd.read_csv("./data/urlhaus_data.csv", skiprows=8)  # Skip metadata rows

def contains_suspicious_keywords(url):
    """Check if the URL contains suspicious keywords."""
    url_lower = url.lower()
    return any(word in url_lower for word in SUSPICIOUS_KEYWORDS)

def matches_lookalike_pattern(domain):
    """Check if the domain matches known phishing lookalike patterns."""
    for pattern in LOOKALIKE_PATTERNS:
        if re.search(pattern, domain):
            return True
    return False

def has_risky_tld(domain):
    """Check if the domain has a risky TLD."""
    tld = domain.split(".")[-1]
    return tld in RISKY_TLDS

def check_malicious(url, phishtank_df, urlhaus_df):
    """Check if the URL is in PhishTank, URLHaus, or contains suspicious elements."""
    parsed_url = urlparse(url)
    domain = parsed_url.netloc

    if url in phishtank_df["url"].values:
        return "phishing"
    if url in urlhaus_df["url"].values:
        return "malware"
    if contains_suspicious_keywords(url):
        return "suspicious"
    if matches_lookalike_pattern(domain):
        return "lookalike"
    if has_risky_tld(domain):
        return "risky_tld"
    return "safe"

def analyze_urls():
    """Analyze URLs extracted from QR codes and classify threats."""
    df = pd.read_csv("./data/qr_extracted_urls.csv")
    phishtank_df = fetch_phishtank_data()
    urlhaus_df = fetch_urlhaus_data()

    df["threat_type"] = df["URL"].apply(lambda x: check_malicious(x, phishtank_df, urlhaus_df))
    df.to_csv("./data/qr_analyzed_urls.csv", index=False)
    return "URLs analyzed and labeled successfully."

if __name__ == "__main__":
    print(analyze_urls())