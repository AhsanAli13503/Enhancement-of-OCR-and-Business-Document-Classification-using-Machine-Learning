import pyap

def find_adress(data):
    for abc in data:
        addresses = pyap.parse(abc, country='US')
        for adres in addresses:
            return adres
    return "no adress in image"