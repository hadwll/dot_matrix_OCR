from datetime import datetime
import numpy as np

def interpret_date(received_date, max_days):
    
    ##convert to integer format
    received_date = list(map(int,received_date))
    year = received_date[:4]
    str_year = [str(year) for year in year]
    a_year = "".join(str_year)
    int_year = int(a_year)
    
    month =received_date[5:7]
    str_month = [str(month) for month in month]
    a_month = "".join(str_month)
    int_month = int(a_month)
    
    day =  received_date[8:]
    str_day = [str(day) for day in day]
    a_day = "".join(str_day)
    int_day = int(a_day)
    
    
    date = [int_year ,int_month, int_day]
    dt = datetime(*date)
    
    #print(dt)
    #print(datetime.now())
    
    now = datetime.now()
    pack = dt
    delta = now - pack
    
    if abs(delta.days) > max_days:
        return(False)

    else:
        return(True)
        
received_date =  ['2', '0', '1', '9', '2', '0', '3', '2', '3', '0']

print(interpret_date(received_date,90))