import requests


def extract_filename_from_url(url):
    try:
        end = url[url.rfind("/") + 1 :]
        filename = end[: end.find("?")]
        return filename
    except:
        return None


def download_from_dropbox(url, outfile=None):
    url = url.replace("?dl=0", "")
    url = url.replace("?dl=1", "")
    if not "?raw=1" in url:
        url += "?raw=1"
    r = requests.get(url, allow_redirects=True)
    if not outfile:
        outfile = extract_filename_from_url(url)
    if outfile:
        open(outfile, "wb").write(r.content)
