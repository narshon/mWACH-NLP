# -*- coding: utf-8 -*-
"""
Created on Mon May 13 19:08:26 2019

@author: Eunice mbeyu
"""

import pandas as pd
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pylab import *
import datetime
import matplotlib.dates as mdates

messages = pd.read_csv("messages.csv",  encoding = "ISO-8859-1")
participant_details = pd.read_csv("participant_details.csv",  encoding = "ISO-8859-1")
length_of_stay = pd.read_csv("outputs/data/length_of_stay.csv",  encoding = "ISO-8859-1")
messages_summary = pd.read_csv("outputs/data/messages_summary.csv",  encoding = "ISO-8859-1")
part_xteristics = pd.read_csv("outputs/data/part_xteristics.csv",  encoding = "ISO-8859-1")
participant_summary = pd.read_csv("outputs/data/participant_summary.csv",  encoding = "ISO-8859-1")
system_msgs_summary = pd.read_csv("outputs/data/system_msgs_summary.csv",  encoding = "ISO-8859-1")
topic_summary = pd.read_csv("outputs/data/topic_summary.csv",  encoding = "ISO-8859-1")

#Total messages --pie
labels = 'System', 'Nurse', 'Participant'
sizes = [messages_summary[["total_system_msg"]].iloc[0].values[0], messages_summary[["total_nurse_msg"]].iloc[0].values[0], messages_summary[["total_participant_msg"]].iloc[0].values[0] ]   
colors = ['gold', 'lightgreen', 'lightskyblue']
explode = (0, 0, 0.1)  # explode 1st slice

# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=140)

plt.axis('equal')
plt.title("All Messages (n=59506)")
plt.show()

#Total messages by Topic -- bar
performance = [messages_summary[["adherence"]].iloc[0].values[0], messages_summary[["antenatal_concerns"]].iloc[0].values[0], messages_summary[["delivery_concerns"]].iloc[0].values[0], 
              messages_summary[["family_planning"]].iloc[0].values[0], messages_summary[["immunization"]].iloc[0].values[0], messages_summary[["infant_health"]].iloc[0].values[0], messages_summary[["other"]].iloc[0].values[0], messages_summary[["side_effects"]].iloc[0].values[0], messages_summary[["validation"]].iloc[0].values[0], messages_summary[["visits"]].iloc[0].values[0]]   
objects = ['Adherence', 'antenatal_concerns', 'delivery_concerns','family_planning','immunization','infant_health','other','side_effects','validation','visits']
x_axis = ["A","B","C","D","E","F","G","H","I","J",]
y_pos = np.arange(len(objects))

plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, x_axis)
plt.ylabel('Messages')
plt.title('Topics')
patch1 = mpatches.Patch(label='A - adherence')
patch2 = mpatches.Patch(label='B - antenatal_concerns')
patch3 = mpatches.Patch(label='C - delivery_concerns')
patch4 = mpatches.Patch(label='D - family_planning')
patch5 = mpatches.Patch(label='E - immunization')
patch6 = mpatches.Patch(label='F - infant_health')
patch7 = mpatches.Patch(label='G - other')
patch8 = mpatches.Patch(label='H - side_effects')
patch9 = mpatches.Patch(label='I - validation')
patch10 = mpatches.Patch(label='J - visits')
plt.legend(handles=[patch1, patch2,patch3,patch4,patch5,patch6,patch7,patch8,patch9,patch10])
plt.show()

#Participants enrolment
import matplotlib.pyplot as pyplt
from matplotlib.dates import DateFormatter

date_enrolled = participant_details['Enrolled']
study_id = participant_details['Study ID'].values

fig, ax = pyplt.subplots()
#date_enrolled = date_enrolled.date.astype('O')

ax.plot(date_enrolled, study_id)

myFmt = DateFormatter("%Y-%m")
ax.xaxis.set_major_formatter(myFmt)

## Rotate date labels automatically
#fig.autofmt_xdate()
# pyplt.plot(date_enrolled, study_id)
pyplt.show()

#Participant length of stay
pid = participant_summary['pid'].values
study_wk = participant_summary['study_wk'].values
plt.xlabel('Participants')
plt.ylabel('Duration of Stay (wks)')
plt.title('Participant length of stay')
pyplt.scatter(pid, study_wk)
pyplt.show()
# save to file
pyplt.savefig('outputs/plots/length_of_stay.png')

#Topics

adherence = topic_summary['adherence'].values
antenatal_concerns = topic_summary['antenatal_concerns'].values
delivery_concerns = topic_summary['delivery_concerns'].values
family_planning = topic_summary['family_planning'].values
immunization = topic_summary['immunization'].values
infant_health = topic_summary['infant_health'].values
other = topic_summary['other'].values
side_effects = topic_summary['side_effects'].values
validation = topic_summary['validation'].values
visits = topic_summary['visits'].values
study_wk = topic_summary['study_wk'].values
plt.xlabel('Study Wk')
plt.ylabel('Messages (Per Person per Wk)')
plt.title('Messages By Topic in Wks')
pyplt.figure(num=1, figsize=(16, 10), dpi=120, facecolor='w', edgecolor='k')
pyplt.plot(study_wk, adherence, 'c')
pyplt.plot(study_wk, antenatal_concerns, 'r')
pyplt.plot(study_wk, delivery_concerns, 'b')
pyplt.plot(study_wk, family_planning, 'g')
pyplt.plot(study_wk, immunization, 'w')
pyplt.plot(study_wk, infant_health, 'k')
pyplt.plot(study_wk, other, 'm')
pyplt.plot(study_wk, side_effects, 'c')
pyplt.plot(study_wk, validation, 'w')
pyplt.plot(study_wk, visits, 'y')


patch1 = mpatches.Patch('c', label='A - adherence')
patch2 = mpatches.Patch('r',label='B - antenatal_concerns')
patch3 = mpatches.Patch('b',label='C - delivery_concerns')
patch4 = mpatches.Patch('g',label='D - family_planning')
patch5 = mpatches.Patch('w',label='E - immunization')
patch6 = mpatches.Patch('k', label='F - infant_health')
patch7 = mpatches.Patch('m', label='G - other')
patch8 = mpatches.Patch('c', label='H - side_effects')
patch9 = mpatches.Patch('w', label='I - validation')
patch10 = mpatches.Patch('y', label='J - visits')
pyplt.legend(handles=[patch1, patch2,patch3,patch4,patch5,patch6,patch7,patch8,patch9,patch10])
pyplt.show()
# save to file
pyplt.savefig('outputs/plots/topics_per_wk.png')

#Topics By edd_date
adherence = topic_summary['adherence'].values
antenatal_concerns = topic_summary['antenatal_concerns'].values
delivery_concerns = topic_summary['delivery_concerns'].values
family_planning = topic_summary['family_planning'].values
immunization = topic_summary['immunization'].values
infant_health = topic_summary['infant_health'].values
other = topic_summary['other'].values
side_effects = topic_summary['side_effects'].values
validation = topic_summary['validation'].values
visits = topic_summary['visits'].values
edd_wk = topic_summary['edd_wk'].values
plt.xlabel('EDD Wk')
plt.ylabel('Messages (Per Person per Wk)')
plt.title('Messages By Topic in Edd Wks')
pyplt.figure(num=1, figsize=(16, 10), dpi=120, facecolor='w', edgecolor='k')
pyplt.plot(edd_wk, adherence, 'c')
pyplt.plot(edd_wk, antenatal_concerns, 'r')
pyplt.plot(edd_wk, delivery_concerns, 'b')
pyplt.plot(edd_wk, family_planning, 'g')
pyplt.plot(edd_wk, immunization, 'w')
pyplt.plot(edd_wk, infant_health, 'k')
pyplt.plot(edd_wk, other, 'm')
pyplt.plot(edd_wk, side_effects, 'c')
pyplt.plot(edd_wk, validation, 'w')
pyplt.plot(edd_wk, visits, 'y')


patch1 = mpatches.Patch('c', label='A - adherence')
patch2 = mpatches.Patch('r',label='B - antenatal_concerns')
patch3 = mpatches.Patch('b',label='C - delivery_concerns')
patch4 = mpatches.Patch('g',label='D - family_planning')
patch5 = mpatches.Patch('w',label='E - immunization')
patch6 = mpatches.Patch('k', label='F - infant_health')
patch7 = mpatches.Patch('m', label='G - other')
patch8 = mpatches.Patch('c', label='H - side_effects')
patch9 = mpatches.Patch('w', label='I - validation')
patch10 = mpatches.Patch('y', label='J - visits')
pyplt.legend(handles=[patch1, patch2,patch3,patch4,patch5,patch6,patch7,patch8,patch9,patch10])
pyplt.show()
# save to file
pyplt.savefig('outputs/plots/topics_per_wk.png')

#Participants trajectory - All
msgs_all = participant_summary['total_msgs'].values
study_wk = participant_summary['study_wk'].values
plt.xlabel('Study Wk')
plt.ylabel('All Messages')
plt.title('Weekly Msgs Per Participant (ALL)')
plt.scatter(study_wk, msgs_all)
pyplt.show()

#Participants trajectory - Participant msgs
msgs_all = participant_summary['participant_msgs'].values
study_wk = participant_summary['study_wk'].values
plt.xlabel('Study Wk')
plt.ylabel('All Messages')
plt.title('Weekly Msgs Per Participant (Participant msgs)')
plt.scatter(study_wk, msgs_all, color='green')
pyplt.show()

#Participants trajectory - System msgs
msgs_system = participant_summary['system_msgs'].values
study_wk = participant_summary['study_wk'].values
plt.xlabel('Study Wk')
plt.ylabel('All Messages')
plt.title('Weekly Msgs Per Participant (System msgs)')
plt.scatter(study_wk, msgs_system, color='red')
pyplt.show()

#Participants trajectory - Nurse msgs
msgs_nurse = participant_summary['nurse_msgs'].values
study_wk = participant_summary['study_wk'].values
plt.xlabel('Study Wk')
plt.ylabel('All Messages')
plt.title('Weekly Msgs Per Participant (Nurse msgs)')
plt.scatter(study_wk, msgs_nurse,color='skyblue')
pyplt.show()

#Participants EDD trajectory - All
msgs_all = messages['mid'].values
edd_wk = messages['edd_wk'].values
plt.xlabel('EDD Wk')
plt.ylabel('All Messages')
plt.title('Messaging vs EDD (ALL)')
plt.scatter(edd_wk, msgs_all)
pyplt.show()

#Participants Engagements - System
msgs_all =messages[messages['sent_by']=="system"].loc[:,'mid'].values
pid =    messages[messages['sent_by']=="system"].loc[:,'pid'].values
plt.xlabel('Participants')
plt.ylabel('All Messages')
plt.title('Participants Engagement (SYSTEM Msgs)')
plt.plot(pid, msgs_all, color='red')
pyplt.show()

#Participants Engagements - Participant
msgs_all =messages[messages['sent_by']=="participant"].loc[:,'mid'].values
pid =    messages[messages['sent_by']=="participant"].loc[:,'pid'].values
plt.xlabel('Participants')
plt.ylabel('All Messages')
plt.title('Participants Engagement (SYSTEM Msgs)')
plt.scatter(pid, msgs_all, color='green')
pyplt.show()

#Participants Engagements - Nurse
msgs_all =messages[messages['sent_by']=="nurse"].loc[:,'mid'].values
pid =    messages[messages['sent_by']=="nurse"].loc[:,'pid'].values
plt.xlabel('Participants')
plt.ylabel('All Messages')
plt.title('Participants Engagement (Nurse Msgs)')
plt.scatter(pid, msgs_all, color='skyblue')
pyplt.show()

#Total messages --pie
labels = 'Has Response', "No Response"
sizes = [len(system_msgs_summary[system_msgs_summary['has_response']==1].loc[:,'pid'].values), len(system_msgs_summary[system_msgs_summary['has_response']==0].loc[:,'pid'].values)]   
colors = ['green', 'red']
explode = (0, 0)  # explode 1st slice
# Plot
plt.pie(sizes, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')
plt.title("All Messages (n=26021)")
plt.show()

#System Messages -All in threads 
msgs_total = system_msgs_summary['total_in_thread'].values
msgs_part = system_msgs_summary['msgs_from_participant'].values
msgs_nurse = system_msgs_summary['msgs_from_nurse'].values
study_wk =    system_msgs_summary['study_wk'].values
plt.xlabel('Study Wk')
plt.ylabel('All Feedback')
plt.title('Messages in Threads by Study Wk (Nurse Feedback)')
#plt.scatter(study_wk, msgs_total) #, 
#plt.scatter(study_wk, msgs_nurse, color='skyblue')
plt.scatter(study_wk, msgs_nurse, color='skyblue')
pyplt.show()





