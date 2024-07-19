import os
import csv
from nanpy import (ArduinoApi, SerialManager)
from time import sleep
import matplotlib.pyplot as plt
import termcolor as tm
import statistics as st
import numpy as np
import smtplib
import pandas as pd
from picamera import PiCamera
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from email.mime.text import MIMEText
import datetime as dt
import seaborn as sns

import RPi.GPIO as GPIO
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)
GPIO.setup(8, GPIO.OUT, initial=GPIO.LOW)
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)
GPIO.setup(3, GPIO.OUT, initial=GPIO.LOW)
###############################################1

cputemp=[]
critical=[]
harsh=[]
normal=[]

sound_phase1=[]
temp_phase1=[]
gas_phase1=[]
magnet_phase1=[]
uv_phase1=[]
vib_phase1=[]


sound_phase2=[]
temp_phase2=[]
gas_phase2=[]
magnet_phase2=[]
uv_phase2=[]
vib_phase2=[]

sound_AVR=[]
#temp_AVR=[]
gas_AVR=[]
magnet_AVR=[]
uv_AVR=[]
vib_AVR=[]




date_time=[]
env_co=[]
envdata=[]
avrdata=[]
org_val=[]
pred_val=[]
#############################################2
analogPin0 = 0  # sound
analogPin1 = 1  # gas
analogPin2 = 2  # magnetic
analogPin3 = 3  # uv sense
analogPin4 = 4  # vibration sense
ledpin=8
ledState=False
conn=SerialManager()
a=ArduinoApi(connection=conn)
a.pinMode(ledpin,a.OUTPUT)
#############################################3
camera = PiCamera()
nos=int(input("enter the number of sample"))
##############################################
for r in range(1,nos):
    temp = os.popen("vcgencmd measure_temp").readline()
    val0 = a.analogRead(analogPin0)
    val0a = a.analogRead(analogPin0)
    
    val1 = a.analogRead(analogPin1)
    val1a = a.analogRead(analogPin1)
    
    val2 = a.analogRead(analogPin2)
    val2a = a.analogRead(analogPin2)
    
    val3 = a.analogRead(analogPin3)
    val3a = a.analogRead(analogPin3)
    
    val4 = a.analogRead(analogPin4)
    val4a = a.analogRead(analogPin4)
#################################################333
    sound_phase1.append((val0+83.2073)/(11.003*10))
    gas_phase1.append((5-((val1/1024)*5))/((val1/1024)*5))
    magnet_phase1.append((val2)/100)
    uv_phase1.append(val3/100)
    vib_phase1.append(val4/100)

#####################################################
    sound_phase2.append((val0a+83.2073)/(11.003*10))
    gas_phase2.append((5-((val1a/1024)*5))/((val1a/1024)*5))
    magnet_phase2.append((val2a)/100)
    uv_phase2.append(val3a/100)
    vib_phase2.append(val4a/100)
    cputemp.append(temp)


    
    datedata=dt.datetime.now().strftime('%H:%M:%S')
    date_time.append(datedata)

sensors_data=pd.DataFrame(data={"DATE/time":date_time,"1.SOUND phase1": sound_phase1,"2.Gas phase1": gas_phase1,"3.magnetic phase1": magnet_phase1,
                                "4.UV phase1": uv_phase1,"5.Vibration phase1":vib_phase1,"6.SOUND phase2": sound_phase2,"7.Gas phase2": gas_phase2,
                                "8.magnetic phase2": magnet_phase2,"9.UV phase2": uv_phase2,"10.Vibration phase2":vib_phase2 })
sensors_data.to_csv("/home/pi/FINAL_PROJECT_CODE/sensors_data_phase_condition_phase_states_CN.csv",sep=",",index=False)
data_new=pd.read_csv("/home/pi/FINAL_PROJECT_CODE/sensors_data_phase_condition_phase_states_CN.csv")
print(data_new.head())


print("-----------------------------------------NEXT STAGE---------------------------------------------------")

data_sense = data_new   .iloc[:, [1,2,3,4,5,6,7,8,9,10]].values
print("Total DATA=",len(data_sense))
data_sense1=np.asarray(data_sense)

sound_phase1_cond=[]
uv_phase1_cond=[]
gas_phase1_cond=[]
magn_phase1_cond=[]
vib_phase1_cond=[]

sound_phase2_cond=[]
uv_phase2_cond=[]
gas_phase2_cond=[]
magn_phase2_cond=[]
vib_phase2_cond=[]
print('---------------------PHASE1----------------')

uv_thresh1=data_new["1.SOUND phase1"].mean()
print(uv_thresh1)

sound_thresh1=data_new["2.Gas phase1"].mean()
print(sound_thresh1)

magn_thresh1=data_new["3.magnetic phase1"].mean()
print(magn_thresh1)

vib_thresh1=data_new["4.UV phase1"].mean()
print(vib_thresh1)

gas_thresh1=data_new["5.Vibration phase1"].mean()
print(gas_thresh1)
##########################################################
print("-------------------PHASE2--------------------")
uv_thresh2=data_new["6.SOUND phase2"].mean()
print("uvthresh=",uv_thresh2)

sound_thresh2=data_new["7.Gas phase2"].mean()
print("SOUNDTHRESH=",sound_thresh2)

magn_thresh2=data_new["8.magnetic phase2"].mean()
print('MAGNETTHRUSH=',magn_thresh2)

vib_thresh2=data_new["9.UV phase2"].mean()
print('vIBRATION_THRESH=',vib_thresh2)

gas_thresh2=data_new["10.Vibration phase2"].mean()
print('GAS THRESH=',gas_thresh2)


for s in range(len(data_new)):
    if sound_thresh1 >= data_sense1[s,0]:
        sound_phase1_cond.append('N')
    else:
        sound_phase1_cond.append('C')
##############################################
for h in range(len(data_new)):
    if data_sense1[h,1] <= gas_thresh1:
        gas_phase1_cond.append('N')
    else:
        gas_phase1_cond.append('C')
##############################################3
for m in range(len(data_new)):
    if data_sense1[m,2] <= magn_thresh1:
        magn_phase1_cond.append('N')
    else:
        magn_phase1_cond.append('C')

################################################
for l in range(len(data_new)):
    if data_sense1[l,3] <= uv_thresh1:
        uv_phase1_cond.append('N')
    else:
        uv_phase1_cond.append('C')
###############################################3
for v in range(len(data_new)):
    if data_sense1[v,4] <= vib_thresh1:
        vib_phase1_cond.append('N')
    else:
        vib_phase1_cond.append('C')



for sp in range(len(data_new)):
    if sound_thresh2 >= data_sense1[sp,5]:
        sound_phase2_cond.append('N')
    else:
        sound_phase2_cond.append('C')
##############################################
for he in range(len(data_new)):
    if data_sense1[he,6] <= gas_thresh2:
        gas_phase2_cond.append('N')
    else:
        gas_phase2_cond.append('C')
##############################################3
for me in range(len(data_new)):
    if data_sense1[me,7] <= magn_thresh2:
        magn_phase2_cond.append('N')
    else:
        magn_phase2_cond.append('C')

################################################
for le in range(len(data_new)):
    if data_sense1[le,8] <= uv_thresh2:
        uv_phase2_cond.append('N')
    else:
        uv_phase2_cond.append('C')
###############################################3
for ve in range(len(data_new)):
    if data_sense1[ve,9] <= vib_thresh2:
        vib_phase2_cond.append('N')
    else:
        vib_phase2_cond.append('C')

data_son = pd.read_csv("/home/pi/FINAL_PROJECT_CODE/sensors_data_phase_condition_phase_states_CN.csv")
data_son['11.S1P1C1'] = sound_phase1_cond
data_son['12.S2P1C1'] = uv_phase1_cond
data_son['13.S3P1C1'] = gas_phase1_cond
data_son['14.S4P1C1'] = magn_phase1_cond
data_son['15.S5P1C1'] = vib_phase1_cond

############################
data_son['16.S1P2C1'] = sound_phase2_cond
data_son['17.S2P2C1'] = uv_phase2_cond
data_son['18.S3P2C1'] = gas_phase2_cond
data_son['19.S4P2C1'] = magn_phase2_cond
data_son['20.S5P2C1'] = vib_phase2_cond
data_son.to_csv("/home/pi/FINAL_PROJECT_CODE/sensors_data_phase_condition_phase_states_CN.csv", index=False)
data_phase_cond=pd.read_csv("/home/pi/FINAL_PROJECT_CODE/sensors_data_phase_condition_phase_states_CN.csv")
data_phase_cond_values=data_phase_cond.iloc[:,[11,12,13,14,15,16,17,18,19,20]].values
print(data_phase_cond.head())

phase1_cond=[]
phase2_cond=[]
count_cond_phase1=[]
count_cond_phase2=[]
#COUNT NO OF Cs And Ns and compare the situatoons make phase1 and phase state
phase_test=np.asarray(data_phase_cond_values)
print(phase_test)
for g in range(len(phase_test)):
    setph1=phase_test[g,[0,1,2,3,4]]
    count_cond_phase1.append(setph1)
    if list(count_cond_phase1[g]).count('C') in range(3,6):
        phase1_cond.append("C")
    else:
        phase1_cond.append('N')

for o in range(len(phase_test)):
    setph2=phase_test[o,[5,6,7,8,9]]
    count_cond_phase2.append(setph2)
    if list(count_cond_phase2[o]).count('C') in range(3,6):
        phase2_cond.append("C")
    else:
        phase2_cond.append('N')

data_phase_cond['21.PhaseState1'] = phase1_cond
data_phase_cond['22.PhaseState2'] = phase2_cond
data_phase_cond.to_csv('/home/pi/FINAL_PROJECT_CODE/sensors_data_phase_condition_phase_states_CN.csv', index=False)

final_state_data=pd.read_csv("/home/pi/FINAL_PROJECT_CODE/sensors_data_phase_condition_phase_states_CN.csv")
print(final_state_data.head())

phase_state_condition1=list(final_state_data["21.PhaseState1"])
print("phase_conditioin1-->",phase_state_condition1)

phase_state_condition2=list(final_state_data["22.PhaseState2"])
print("phase_condition2-->",phase_state_condition2)

stateTF=[]

for t in range(len(phase_state_condition1)):
    if phase_state_condition2[t] == phase_state_condition1[t]=='C':
        stateTF.append('C')
        '''
        camera.start_preview()
        camera.resolution=(1280,720)
        camera.framerate=30
        sleep(2)
        camera.capture('/home/pi/FINAL_PROJECT_CODE/harsh1.jpg')
        camera.stop_preview()
    
        sleep(2)
                       
        camera.start_preview()
        camera.resolution=(1280,720)
        camera.framerate=30
        camera.color_effects = (128,128)
        camera.capture('/home/pi/FINAL_PROJECT_CODE/harsh2.jpg')
        camera.stop_preview()
        sleep(1)
        mail_content = """  HELP HELP HELP YOUR DEVICE RUN INTO CRITICAL MODE """
    #The mail addresses and password
        sender_address ='  '
        sender_pass =' '
        receiver_address =' '
    #Setup the MIME
        message = MIMEMultipart()
        message['From'] = sender_address
        message['To'] = receiver_address
        message['Subject'] = 'DEVICE RUN TO CRITICAL CONDITION.'   #The subject line

        with open('/home/pi/FINAL_PROJECT_CODE/harsh1.jpg', 'rb') as fp:
                dataimage = fp.read()
        image1 = MIMEImage(dataimage)
        message.attach(image1)

        with open('/home/pi/FINAL_PROJECT_CODE/harsh2.jpg', 'rb') as ft:
                dataimage1 = ft.read()
        image2 = MIMEImage(dataimage1)
        message.attach(image2)

    #The body and the attachments for the mail
        message.attach(MIMEText(mail_content, 'plain'))
    #Create SMTP session for sending the mail
        session = smtplib.SMTP('smtp.gmail.com', 587) #use gmail with port
        session.starttls() #enable security
        session.login(sender_address, sender_pass) #login with mail_id and password
        text = message.as_string()
        session.sendmail(sender_address, receiver_address, text)
        session.quit()
        print("AUTO SERVICE MSSG SENT TO DEPARTMENT REGARDING CRITICAL ENVIORMENT")
        from twilio.rest import Client

        # Your Account SID from twilio.com/console
        account_sid = 'AC6be71b264ecd8aaae99c24f514a48dd0'
# Your Auth Token from twilio.com/console
        auth_token  = '4de8e9ca015cf60518a713c5e7454e7b'

        client = Client(account_sid, auth_token)

        message = client.messages.create(body="Message from device Please check email for complete report, or make a call to your friend as deviec run into emergency condition",
        to='whatsapp:+353894864958',
        from_='whatsapp:+14155238886')

        print(message.sid)
        '''

    else:
        stateTF.append("N")
        
print(stateTF)
final_state_data['23.FSD-T/F']=stateTF
final_state_data.to_csv('/home/pi/FINAL_PROJECT_CODE/sensors_data_phase_condition_phase_states_CN.csv', index=False)

#######################################################################################################################################

# Support Vector Machine (SVM)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('/home/pi/FINAL_PROJECT_CODE/sensors_data_phase_condition_phase_states_CN.csv')
print(dataset.head())
X = dataset.iloc[:, [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]].values
#X = dataset.iloc[:, [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]].values
#X=np.asarray(X)
print(X.shape)
print(X)
print("X",len(X))
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:,10] = labelencoder_X.fit_transform(X[:,10])
X[:,11] = labelencoder_X.fit_transform(X[:,11])
X[:,12] = labelencoder_X.fit_transform(X[:,12])
X[:,13] = labelencoder_X.fit_transform(X[:,13])
X[:,14] = labelencoder_X.fit_transform(X[:,14])
X[:,15] = labelencoder_X.fit_transform(X[:,15])
X[:,16] = labelencoder_X.fit_transform(X[:,16])
X[:,17] = labelencoder_X.fit_transform(X[:,17])
X[:,18] = labelencoder_X.fit_transform(X[:,18])
X[:,19] = labelencoder_X.fit_transform(X[:,19])
X[:,20] = labelencoder_X.fit_transform(X[:,20])
X[:,21] = labelencoder_X.fit_transform(X[:,21])
X=X[:,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]]
print("Encoded value-->",X)


y = dataset.iloc[:, [-1]].values
y=np.asarray(y)
y=y.ravel()
print(y.shape)
print(y)
print("y",len(y))

from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

print(len(X_train))
print(X_train)
print(len(y_train))
print(y_train)


# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

print("Normal condition count=",list(y_pred).count(1))
print("Critical condition count=",list(y_pred).count(0))


print("Predictions-->",y_pred)
print("Normal condition count=",list(y_pred).count(1))
normal=list(y_pred).count(1)
normla_per=(normal/len(y_pred))*100
print("Normal condition percentage-->",normla_per,"%")


print("Critical condition count=",list(y_pred).count(0))
crit=list(y_pred).count(0)
crit_per=(crit/len(y_pred))*100
print("Critical condition percentage-->",crit_per,"%")

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)

acc_score=accuracy_score(y_pred,y_test)
print("ACCURACY=",acc_score*100,"%")

if normla_per in range(90,101):
    
    
    GPIO.output(8,GPIO.HIGH)
    sleep(3)
    GPIO.output(8,GPIO.LOW)
    
if normla_per in range(60,90):
    GPIO.output(8,GPIO.HIGH)
    GPIO.output(3,GPIO.HIGH)
    sleep(3)
    GPIO.output(8,GPIO.LOW)
    GPIO.output(3,GPIO.LOW)

if normla_per in range(0,60):
    
    GPIO.output(3,GPIO.HIGH)
    sleep(3)
    GPIO.output(3,GPIO.LOW)
    
    camera.start_preview()
    camera.resolution=(1280,720)
    camera.framerate=30
    sleep(2)
    camera.capture('/home/pi/FINAL_PROJECT_CODE/harsh1.jpg')
    camera.stop_preview()
    
    sleep(2)
        
    camera.start_preview()
    camera.resolution=(1280,720)
    camera.framerate=30
    camera.color_effects = (128,128)
    camera.capture('/home/pi/FINAL_PROJECT_CODE/harsh2.jpg')
    camera.stop_preview()
    sleep(1)
    mail_content = """  HELP HELP HELP YOUR DEVICE RUN INTO CRITICAL MODE """
    #The mail addresses and password
    sender_address ='thesisyashsharma@gmail.com'
    sender_pass ='@W20087278'
    receiver_address ='thesisyashsharma@gmail.com'
    #Setup the MIME
    message = MIMEMultipart()
    message['From'] = sender_address
    message['To'] = receiver_address
    message['Subject'] = 'DEVICE RUN TO CRITICAL CONDITION.'   #The subject line

    with open('/home/pi/FINAL_PROJECT_CODE/harsh1.jpg', 'rb') as fp:
            dataimage = fp.read()
    image1 = MIMEImage(dataimage)
    message.attach(image1)

    with open('/home/pi/FINAL_PROJECT_CODE/harsh2.jpg', 'rb') as ft:
            dataimage1 = ft.read()
    image2 = MIMEImage(dataimage1)
    message.attach(image2)

    #The body and the attachments for the mail
    message.attach(MIMEText(mail_content, 'plain'))
    #Create SMTP session for sending the mail
    session = smtplib.SMTP('smtp.gmail.com', 587) #use gmail with port
    session.starttls() #enable security
    session.login(sender_address, sender_pass) #login with mail_id and password
    text = message.as_string()
    session.sendmail(sender_address, receiver_address, text)
    session.quit()
    print("AUTO SERVICE MSSG SENT TO DEPARTMENT REGARDING CRITICAL ENVIORMENT")
    from twilio.rest import Client

        # Your Account SID from twilio.com/console
    account_sid = 'AC6be71b264ecd8aaae99c24f514a48dd0'
# Your Auth Token from twilio.com/console
    auth_token  = '4de8e9ca015cf60518a713c5e7454e7b'

    client = Client(account_sid, auth_token)

    message = client.messages.create(body="Message from device Please check email for complete report, or make a call to your friend as deviec run into emergency condition",
    to='whatsapp:+353894864958',
    from_='whatsapp:+14155238886')

    print(message.sid)

plt.plot(X[:,[1,2,3,4,5,6,7,8,9,10]])

         
# Visualising the Training set results
from matplotlib.colors import ListedColormap
plt.figure(1)
#plt.subplot(1,1,1)
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
Xpred = np.array([X1.ravel(), X2.ravel()] + [np.repeat(0, X1.ravel().size) for _ in range(19)]).T
# Xpred now has a grid for x1 and x2 and average value (0) for x3 through x13
pred = classifier.predict(Xpred).reshape(X1.shape)   # is a matrix of 0's and 1's !
plt.contourf(X1, X2, pred,alpha = 0.75, cmap = ListedColormap(('silver', 'white')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('LEARNING PATTERN GREEN-NORMAL RED-CRITICAL')
plt.xlabel('ALL PARAMETER')
plt.ylabel('TARGET')
plt.legend()
#plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
#plt.subplot(2,1,2)
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
Xpred = np.array([X1.ravel(), X2.ravel()] + [np.repeat(0, X1.ravel().size) for _ in range(19)]).T
# Xpred now has a grid for x1 and x2 and average value (0) for x3 through x13
pred = classifier.predict(Xpred).reshape(X1.shape)   # is a matrix of 0's and 1's !
plt.contourf(X1, X2, pred,alpha = 0.75, cmap = ListedColormap(('silver', 'white')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('PREDICTION PATTERN GREEN-NORMAL RED-CRITICAL ')
plt.xlabel('ALL PARAMETER')
plt.ylabel('TARGET')
plt.legend()
plt.show(block=False)


##################################################################################################


###############################################################################################################

"""
import plotly

import pandas as pd
import chart_studio.plotly as py
import plotly.graph_objs as go
import numpy as np
from chart_studio.plotly import iplot
#from plotly.offline import download_plotlyjs, plot
from plotly.graph_objs import Scatter, Layout, Figure # plotly graph objects
import time # timer functions
username = 'YashWIT'
api_key = 'guXHSbfFNar7y8RqiXkV'
stream_token = '5y9uw2h96z'
py.sign_in(username, api_key)

#data=pd.read_csv("sensors_two_phase_new_condition.csv")
time=dataset['DATE/time']
sensors1=dataset.iloc[:,[1,2,3,4,5]].values
sensors2=dataset.iloc[:,[6,7,8,9,10]].values

trace1 = go.Scatter(
    x = time,
    y = sensors1,
    name='phase1')

trace2 = go.Scatter(
    x = time,
    y = sensors2,
    name='phase1')

dataset=[trace1,trace2]
layout = Layout(title='Raspberry Pi Streaming Sensor Data')
fig = Figure(data=dataset, layout=layout)
py.plot(fig, filename='Raspberry Pi Streaming Example Values')

stream = py.Stream(stream_token)
stream.open()

"""
