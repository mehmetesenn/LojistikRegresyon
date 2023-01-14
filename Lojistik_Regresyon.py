# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 15:08:03 2023

@author: mehmet esen
"""

import pandas as pd 
import numpy as np

# veri setlerinin proje dahil edilmesi 
Kredi_Tahmini_Train=pd.read_csv("train_kredi_tahmini.csv")
Kredi_Tahmini_Test=pd.read_csv("test_kredi_tahmini.csv")
# bu iki dataframe birleştirilip tüm veri ön işleme
# işlemleri yapıldıktan sonra eğitim ve test kümesi olarak ayırılacaktır.
frame1=[Kredi_Tahmini_Train,Kredi_Tahmini_Test]
Kredi_Tahmini_Orijinal_Veri_Seti=pd.concat(frame1,axis=0)

#kolon isimleri encode yaparken lazım olur
kolon_name=Kredi_Tahmini_Orijinal_Veri_Seti.columns.tolist()
#LoanId kolonu siliniyor
Kredi_Tahmini_Orijinal_Veri_Seti.drop('Loan_ID',axis=1,inplace=True)

#eksik veri içeren kolanların tespiti
eksik_veri_iceren_kolonlar=Kredi_Tahmini_Orijinal_Veri_Seti.columns[Kredi_Tahmini_Orijinal_Veri_Seti.isnull().any()].tolist()


# loan status kolonu hariç diğer kolonlardaki eksik veriler bir alt değer
# ile dolduruldu 
def bosluk_doldur():
    for item in Kredi_Tahmini_Orijinal_Veri_Seti.columns:
        for kolon_name in eksik_veri_iceren_kolonlar:
            if item==kolon_name:
                Kredi_Tahmini_Orijinal_Veri_Seti[item].fillna(method='bfill',inplace=True)
                


bosluk_doldur()

#Loan status eksik veriler doldur
Kredi_Tahmini_Orijinal_Veri_Seti.Loan_Status.fillna(method="ffill",inplace=True)
# eksik veri son durum bakarak veri setindeki 
#nan doldurulduğunu anlayabiliriz eğer değişken size 0 sa eksik
#veriler tüm veri setinde doldurulmuştur.
eksik_veri_Son_Durum=Kredi_Tahmini_Orijinal_Veri_Seti.columns[Kredi_Tahmini_Orijinal_Veri_Seti.isnull().any()].tolist()





# encoding işlemleri one hot encoder ya da label encoder

#encode sürecinde kullanılacak kütüphanelerin dahil edilmesi
from sklearn import preprocessing
#Gender kolonu one-hot encode

gender=Kredi_Tahmini_Orijinal_Veri_Seti[['Gender']].values
gender_ohe=preprocessing.OneHotEncoder()
gender=gender_ohe.fit_transform(gender).toarray()
gender_ohe=pd.DataFrame(gender,columns=["Female","Male"])

#Married Kolonu one-hot encode
married=Kredi_Tahmini_Orijinal_Veri_Seti[['Married']].values
married_ohe=preprocessing.OneHotEncoder()
married=married_ohe.fit_transform(married).toarray()
married_ohe=pd.DataFrame(married,columns=["Married_No","Married_Yes"])

#Education kolonu one-hot encode
education=Kredi_Tahmini_Orijinal_Veri_Seti[['Education']].values
education_ohe=preprocessing.OneHotEncoder()
education=education_ohe.fit_transform(education).toarray()
education_ohe=pd.DataFrame(education,columns=["Graduate","Not Graduate"])

#Self employed one-hot encode
self_employed=Kredi_Tahmini_Orijinal_Veri_Seti[['Self_Employed']].values
self_employed_ohe=preprocessing.OneHotEncoder()
self_employed=self_employed_ohe.fit_transform(self_employed).toarray()
self_employed_ohe=pd.DataFrame(self_employed,columns=["selfemployed_No","selfemployed_Yes"])

#Property-Area one-hot encode
property_area=Kredi_Tahmini_Orijinal_Veri_Seti[['Property_Area']].values
property_area_ohe= preprocessing.OneHotEncoder()
property_area=property_area_ohe.fit_transform(property_area).toarray()
property_area_ohe = pd.DataFrame(property_area, columns=["Rural","Semiurban","Urban"])

#loan status one-hot encode
loan_status=Kredi_Tahmini_Orijinal_Veri_Seti[['Loan_Status']].values
loan_status_ohe= preprocessing.OneHotEncoder()
loan_status=loan_status_ohe.fit_transform(loan_status).toarray()
loan_status_ohe=pd.DataFrame(loan_status,columns=["loan_status_N","loan_status_Y"])

#dataframe birleştirme

frame2=[gender_ohe,married_ohe]
sablon1=pd.concat(frame2,axis=1)
frame3=[sablon1,Kredi_Tahmini_Orijinal_Veri_Seti['Dependents'].to_frame()]
frame3 = [df.reset_index(drop=True) for df in frame3]
sablon2=pd.concat(frame3,axis=1)
frame4=[sablon2,education_ohe]
sablon3=pd.concat(frame4,axis=1)
frame5=[sablon3,self_employed_ohe]
sablon4=pd.concat(frame5,axis=1)
Veri_Seti_5_10=Kredi_Tahmini_Orijinal_Veri_Seti.iloc[:,5:10]
frame6=[sablon4,Veri_Seti_5_10]
frame6 = [df.reset_index(drop=True) for df in frame6]
sablon5=pd.concat(frame6,axis=1)
frame7=[sablon5,property_area_ohe]
sablon6_bagimsiz_degiskenler=pd.concat(frame7,axis=1)

#veri kümesinin eğitim ve test olarak bölünmesi
from sklearn.model_selection import train_test_split
bagimli_degisken=Kredi_Tahmini_Orijinal_Veri_Seti.iloc[:,11:].values
x_train,x_test,y_train,y_test=train_test_split(sablon6_bagimsiz_degiskenler,bagimli_degisken,test_size=0.33,random_state=0)

# standartdizasyon ile öznitelik ölçekleme
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(x_train)
X_test=sc.transform(x_test)


#veri kümesini lojistik regresyon algoritmasına girdi olarak verme

from sklearn.linear_model import LogisticRegression
logr =LogisticRegression(random_state=0)
logr.fit(X_train,y_train)
kredi_durumu_tahmin=logr.predict(X_test)
print(kredi_durumu_tahmin)

#confusion matrix hesaplama 
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,kredi_durumu_tahmin)
print(cm)




#Bu matrisi kullanarak doğruluk oranı hesabı

#Matristeki pozitif tahminlerin sayısını bulun: 122 + 73 = 195
#Matristeki yanlış pozitif tahminlerin sayısını bulun: 63
#Matristeki yanlış negatif tahminlerin sayısını bulun: 66
#Matristeki toplam tahmin sayısını bulun: 195 + 63 + 66 = 324
#Doğruluk oranını hesaplayın: 195 / 324 = %60.12





















