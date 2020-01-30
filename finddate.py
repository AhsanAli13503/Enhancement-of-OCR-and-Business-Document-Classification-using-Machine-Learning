import datefinder
date = None
def date_finder(data):
    for abc in data:
        matches=datefinder.find_dates(abc)
        for match in matches:
            date = match
            return date
    return "no date in image"
