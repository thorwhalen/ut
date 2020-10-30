__author__ = 'thor'


import tldextract


def get_domain(url):
    t = tldextract.extract(url)
    return t.domain


def get_domain_and_suffix(url):
    t = tldextract.extract(url)
    if t.suffix:
        return t.domain + '.' + t.suffix
    else:
        return t.domain


def get_sub_domain_and_suffix(url):
    t = tldextract.extract(url)
    if t.subdomain:
        return t.subdomain + '.' + t.domain + '.' + t.suffix
    else:
        return t.domain + '.' + t.suffix