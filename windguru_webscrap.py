import numpy as np

from getpass import getpass
from selenium import webdriver

from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

import time
from datetime import datetime

import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup

username=("josereis1904")
password=getpass("insert password: ")

driver=webdriver.Chrome("chromedriver.exe")
driver.delete_all_cookies()

today = datetime.now() 

#spots={"Ericeira": "9472","Sagres":"32";"Carcavelos":"1060";"Peniche":"1528";"Nazare":"75856";"Matosinhos":"4167"}
spots={"Carcavelos":"1060"}
      
#dates=[["2008-11-22","2012-12-31"],["2013-01-01","2016-12-31"],["2017-01-01",(str(today.strftime("%Y-%m-%d")))]]
#dates=[["2008-11-22","2012-12-31"],["2013-01-01","2016-12-31"],["2017-01-01","2020-02-29"]]
dates=[["2013-01-01","2016-12-31"]]#,["2013-01-01","2016-12-31"],["2017-01-01","2020-02-29"]]#,["2009-03-30","2009-10-25"],["2009-10-26","2010-03-28"]]
  
for key in spots:
    spot= spots[key]
    for i in dates:
        date1=i[0]
        date2=i[1]
        
        t0 = time.time()
       
        print("Spot ", key, date1,"to", date2, " - Start")
        
        driver.get("https://www.windguru.cz/archive.php?id_spot="+spot+"&id_model=3&date_from="+date1+"&date_to="+date2)
        
        time.sleep(np.random.randint(16,26))
        
        try:
            element=WebDriverWait(driver, np.random.randint(16,46)).until(
                EC.presence_of_element_located((By.XPATH, '//*[@id="sncmp-popup-ok-button"]'))
                )
            element.click()
                
            element=WebDriverWait(driver, np.random.randint(16,46)).until(
                EC.presence_of_element_located((By.XPATH, '//*[@id="inputusername"]'))
                )
            element.send_keys(username)
        
            element=WebDriverWait(driver, np.random.randint(16,36)).until(
                EC.presence_of_element_located((By.XPATH, '//*[@id="jBoxID10"]/div/div[2]/form/label[2]/input'))
                )
            element.send_keys(password)
        
            element=WebDriverWait(driver, np.random.randint(16,26)).until(
                EC.presence_of_element_located((By.XPATH, '//*[@id="jBoxID10"]/div/div[2]/form/button[1]'))
                )
            element.submit()
            
            time.sleep(np.random.randint(3,7))
    
            #Wind direction (turning off)
            element=WebDriverWait(driver, np.random.randint(6,10)).until(
                EC.presence_of_element_located((By.XPATH, '//*[@id="archive_filter"]/fieldset/label[2]'))
                )
            element.click()        
    
            #Gust winds
            """element=WebDriverWait(driver, np.random.randint(6,10)).until(
                EC.presence_of_element_located((By.XPATH, '//*[@id="archive_filter"]/fieldset/label[3]'))
                )
            element.click()"""
            
            #Swell
            element=WebDriverWait(driver, np.random.randint(6,10)).until(
                EC.presence_of_element_located((By.XPATH, '//*[@id="archive_filter"]/fieldset/label[4]'))
                )
            element.click()
            
            #Swell direction
            element=WebDriverWait(driver, np.random.randint(6,10)).until(
                EC.presence_of_element_located((By.XPATH, '//*[@id="archive_filter"]/fieldset/label[5]'))
                )
            element.click()
            
            #Swell period
            element=WebDriverWait(driver, np.random.randint(6,10)).until(
                EC.presence_of_element_located((By.XPATH, '//*[@id="archive_filter"]/fieldset/label[6]'))
                )
            element.click()
            
            #Temperature (Turn off)
            """element=WebDriverWait(driver, np.random.randint(6,10)).until(
                EC.presence_of_element_located((By.XPATH, '//*[@id="archive_filter"]/fieldset/label[7]'))
                )
            element.click()"""
            
            #Rain
            element=WebDriverWait(driver, np.random.randint(6,10)).until(
                EC.presence_of_element_located((By.XPATH, '//*[@id="archive_filter"]/fieldset/label[8]'))
                )
            element.click()
            
            element=WebDriverWait(driver, np.random.randint(6,10)).until(
                EC.presence_of_element_located((By.XPATH, '//*[@id="archive_filter"]/button[1]'))
                )
            element.click()
            
        
            
        finally:
            print("Spot ", key, date1,"to", date2, " - Opening webpage")
            
           
        time.sleep(np.random.randint(45,60))
        html = driver.page_source
        time.sleep(np.random.randint(45,60))
        
        
        df=pd.read_html(html)
            
        driver.quit()
        
        #Initial cleaning   
        
        df1=df[0]    
        
        df1=df1[df1.columns][1:-3]
        
        df1.reset_index(inplace=True, drop=True)
        
        index_todrop=df1[(df1[0].str.contains("GFS 13 km"))].index[2:]
        
        df1.drop(axis=0,index=index_todrop,inplace=True)
        df1.reset_index(inplace=True, drop=True) 
        
        #Saving / open CSV
        df1.to_csv(str("df_"+key+"_"+date1+"-"+date2+".csv"),index=False)
        
        #df1=pd.read_csv("df_Ericeira_2008-11-22-2012-12-31.csv") #####


        def dash(x):
            if( x == "-"):
                ans="nothing"
            else:
                ans="dir"
            return ans
        
        #When trying to include the wind dir. Data is not great
        #dash_from=[1,2,3,4,5,6,7,8,17,18,19,20,21,22,23,24]
        #dash_to=[9,10,11,12,13,14,15,16,25,26,27,28,29,30,31,32]      
                
        dash_from=[9,10,11,12,13,14,15,16]
        dash_to=[17,18,19,20,21,22,23,24]
        for i,ii in zip(dash_to,dash_from):
            df1[i][2:]=df1[ii][2:].apply(dash)
            
            
         #Counters . temporary . to delete   
            
        """xxxx=[] 
        for i in dash_to[0:8]:
            xxxx.append(df1[i].value_counts())
            
        dire=[]
        nothing=[]
        for i in xxxx:
            dire.append(i["dir"])
            nothing.append(i["nothing"])
      
        xxxx2=[] 
        for i in dash_to[8:]:
            xxxx2.append(df1[i].value_counts())
            
        dire2=[]
        for i in xxxx2:
            dire2.append(i["dir"])
            
        print("dire",sum(dire)+sum(dire2))
        print("nothing",sum(nothing))  
        print("expected tabs3:",sum(dire)+sum(dire2))"""

  

            
        
        soup = BeautifulSoup(html, features="lxml")
    
        tabs = soup.find_all('div', attrs= {'id' : 'archive_results'})
        #tabs = soup.find_all('table', attrs= {'class' : 'forecast-ram'})
        #tabs = soup.find('table', {'class':"forecast daily-archive"}).find_all('tr')
        #print(len(tabs))
        
        
        tabs2=(str(tabs)).split("</td>")
        
        df22=df1.copy().T

        
        global tabs3
        tabs3=[]
        for i in tabs2:
            if "transform" in i:
                split=i.split('transform="rotate(')
                tabs3.append(split[1][0:3])
        
        def direction(i):
            global tabs3
            if i=='dir' and len(tabs3) > 0:
                x=tabs3.pop(0)#*i
                return(x)
            else:
                return i
    
    
        df333=df22.applymap(direction)
        df334=df333.T
    
        

        #Data cleaning
                    
        df_clean=pd.DataFrame(df334.copy().T)
        
        #rain fill nan
        df_clean[df_clean.columns][41:48].fillna(value=0,inplace=True)
    
        
        #df3=pd.melt(df2,id_vars=[0,1])
        
        #colss=df1.columns[1:]
        
        #cols=[1, 2, 3, 4, 5, 6, 7, 8, 33, 34, 35, 36, 37, 38, 39, 40]
        
        #df2[cols][2:]=(df2[cols][2:]).astype("int")
        

        df_clean.to_csv(str("df_"+key+"_"+date1+"-"+date2+".csv"),index=False)
        

        print("DataFrame complete")
        

        t1 = time.time()
        total = t1-t0
        print("Total time: "+str(total/60)+"min")
        
        time.sleep(np.random.randint(300,600))
       
    
    
