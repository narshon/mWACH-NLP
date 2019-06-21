# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np

messages = pd.read_csv("messages.csv",  encoding = "ISO-8859-1")
responded = list()
#For system messages, check if the next meesage is a non-system and mark the system message as a response.
for row in messages.itertuples():
    if(row.sent_by == "system"):
      next_msg = messages.iloc[row.Index+1]
      if(next_msg.sent_by == "participant" and row.pid == next_msg.pid ):
          responded.append(1)
      else:
          responded.append(0)
    else:
        responded.append(0)
        
    #print(row.Index, row.pid, row.timestamp) 
        
messages['has_response'] = responded

messages.to_csv("newmsgs.csv")

#total number of participants
    #pid
    #no_study_wks
    #total_msgs
    #system_msgs
    #nurse_msgs
    #participant_msgs
    #messaging_rate
    #messaging_level
no_study_wks=list()
total_msgs=list()
system_msgs = list()
nurse_msgs = list()
participant_msgs = list()
participants = pd.DataFrame({"pid":messages["pid"].unique(),"total_msgs":0})
for row in participants.itertuples():
    df = messages.loc[messages["pid"]==row.pid,"study_wk"].tail(1)
    no_study_wks.append(df.iloc[0])
    total_msgs.append(len(messages.loc[messages["pid"]==row.pid,"study_wk"]))
    pids = messages["pid"]==row.pid
    sys = messages["sent_by"]=="system"
    nurse = messages["sent_by"]=="nurse"
    participant = messages["sent_by"]=="participant"
    system_msgs.append(len(messages[pids & sys]))
    nurse_msgs.append(len(messages[pids & nurse]))
    participant_msgs.append(len(messages[pids & participant]))
    
participants['study_wk'] = no_study_wks 
participants['total_msgs'] = total_msgs
participants['system_msgs'] = system_msgs
participants['nurse_msgs'] = nurse_msgs
participants['participant_msgs'] = participant_msgs

def divided(value1, value2):
    return value1/value2

participants['messaging_rate'] = divided(participants['total_msgs'], participants['study_wk'])

participants.to_csv("outputs/data/participant_summary.csv")

#Messages summary
    #total_count
    #total_system
    #total_nurse
    #total_participants
    # ---- total by topic ------
    #adherence
    #antenatal_concerns
    #delivery_concerns
    #family_planning
    #immunization
    #infant_health
    #other
    #side_effects
    #validation
    #visits
    
msg_summary = list()
total_messages = len(messages)
total_system = len(messages[messages["sent_by"]=="system"])   
total_nurse = len(messages[messages["sent_by"]=="nurse"]) 
total_participant = len(messages[messages["sent_by"]=="participant"]) 

adherence = len(messages[messages["topic"]=="adherence"]) 
antenatal_concerns = len(messages[messages["topic"]=="antenatal-concerns"]) 
delivery_concerns = len(messages[messages["topic"]=="delivery-concenrs"]) 
family_planning = len(messages[messages["topic"]=="family-planing"]) 
immunization = len(messages[messages["topic"]=="immunization"]) 
infant_health = len(messages[messages["topic"]=="infant-health"]) 
other = len(messages[messages["topic"]=="other"]) 
side_effects = len(messages[messages["topic"]=="side-effects"]) 
validation = len(messages[messages["topic"]=="validation"]) 
visits = len(messages[messages["topic"]=="visits"]) 

msgSummary = pd.DataFrame({"total_participants":len(messages["pid"].unique()),"total_messages":total_messages, "total_system_msg":total_system, "total_nurse_msg":total_nurse, "total_participant_msg":total_participant,
                           "adherence":adherence,"antenatal_concerns":antenatal_concerns,"delivery_concerns":delivery_concerns,"family_planning":family_planning,"immunization":immunization,"infant_health":infant_health,"other":other,"side_effects":side_effects,"validation":validation,"visits":visits}
                          , index={0})

msgSummary.to_csv("outputs/data/messages_summary.csv")

#System messages summary
    #system_mid
    #pid
    #external
    #study_wk
    #edd_wk
    #total_in_thread
    #msgs_from_participant
    #msgs_from_nurse
    #has_response
    #has_response_related
    #has_response_not_related
total_in_thread = list()
msgs_from_participant = list()
msgs_from_nurse = list()
related_response = list()
sys_msgs = messages[messages["sent_by"]=="system"]
sys_msgs = sys_msgs[["mid", "pid", "timestamp","external", "study_wk","edd_wk","sent_by","has_response"]] 
for row in sys_msgs.itertuples():
    pid=row.pid
    timestamp_current = row.timestamp
    pids = messages["pid"]==pid
    mids = messages["mid"] > row.mid
    sys = messages["sent_by"]=="system"
    next_sys_msgs = messages.loc[pids & mids & sys,'mid']
    if not next_sys_msgs.empty:
        upper_mid = next_sys_msgs.iloc[0]   #Getting next system message so I can get the upper bound.
    else:
        upper_mid = 67270  #passing the largest mid since the current system message is the last for this participant.
    
    thread_msgs = messages[(messages['mid'] > row.mid) & (messages['mid'] < upper_mid) & (messages["pid"]==pid) & (messages['sent_by'] != "system")]    
    total_in_thread.append(len(thread_msgs))
    parts = thread_msgs["sent_by"]=="participant"
    nurse = thread_msgs["sent_by"]=="nurse"
    msgs_from_participant.append(len(thread_msgs[parts]))
    msgs_from_nurse.append(len(thread_msgs[nurse]))
    #check the response messages if we got related responses to the system message.
    response=thread_msgs["related"]==True
    related_response.append(len(thread_msgs[response & parts]))

    
sys_msgs['total_in_thread'] = total_in_thread   
sys_msgs['msgs_from_participant'] = msgs_from_participant 
sys_msgs['msgs_from_nurse'] = msgs_from_nurse 
sys_msgs['related_response'] = related_response 

#write system messages summary
sys_msgs.to_csv("outputs/data/system_msgs_summary.csv")
  
#Nurses messages summary  *** skipped for now *****
    #pid
    #has_response
    #no_response
    #has_response_related
    #has_response_not_related

#participant messages summary  *** skipped for now *****
    #pid
    #total
    #require_response
    #no_response
    
#message topics summary (participants)  (popular topics according to participants, topics popularity by study_wk, topics popularity by edd,)
    #study_wk
    #pid
    #edd_wk
    #adherence
    #antenatal_concerns
    #delivery_concerns
    #family_planning
    #immunization
    #infant_health
    #other
    #side_effects
    #validation
    #visits
edd_wk = list()
adherence = list()
antenatal_concerns = list()
delivery_concerns = list()
family_planning = list()
immunization = list()
infant_health = list()
other = list()
side_effects = list()
validation = list()
visits = list()
study_wks_data = messages.groupby(['pid','study_wk']).size().reset_index().rename(columns={0:'msgs'}) #getting all study_wk - study_wk is the primary key of this dataset
for row in study_wks_data.itertuples():
    unique_selector = (messages["pid"]==row.pid) & (messages["study_wk"]== row.study_wk)
    edd_wk.append(messages.loc[unique_selector,'edd_wk'].iloc[0])
    adherence.append(len(messages[(messages["topic"]=="adherence") & unique_selector])) 
    antenatal_concerns.append(len(messages[(messages["topic"]=="antenatal-concerns") & unique_selector]) )
    delivery_concerns.append(len(messages[(messages["topic"]=="delivery-concenrs") & unique_selector]) )
    family_planning.append(len(messages[(messages["topic"]=="family-planing") & unique_selector]) )
    immunization.append(len(messages[(messages["topic"]=="immunization") & unique_selector]) )
    infant_health.append(len(messages[(messages["topic"]=="infant-health") & unique_selector]) )
    other.append(len(messages[(messages["topic"]=="other") & unique_selector]) )
    side_effects.append(len(messages[(messages["topic"]=="side-effects") & unique_selector]) )
    validation.append(len(messages[(messages["topic"]=="validation") & unique_selector]) )
    visits.append(len(messages[(messages["topic"]=="visits") & unique_selector]) )

study_wks_data['edd_wk'] = edd_wk 
study_wks_data['adherence'] = adherence 
study_wks_data['antenatal_concerns'] = antenatal_concerns 
study_wks_data['delivery_concerns'] = delivery_concerns 
study_wks_data['family_planning'] = family_planning 
study_wks_data['immunization'] = immunization 
study_wks_data['infant_health'] = infant_health 
study_wks_data['other'] = other 
study_wks_data['side_effects'] = side_effects 
study_wks_data['validation'] = validation 
study_wks_data['visits'] = visits

#write topics summary
study_wks_data.to_csv("outputs/data/topic_summary.csv")

#messages vs length of stay vs vs timing to edd (event)
    #pid
    #study_wk
    #edd_wk
    #participant_msgs
    #nurse_msgs
    #system_msgs
    #total_msgs
edd_wk = list()
participant_msgs = list()
nurse_msgs = list()
system_msgs = list()
lenghtstay_data = messages.groupby(['pid','study_wk']).size().reset_index().rename(columns={0:'total_msgs'}) #getting all study_wk - study_wk is the primary key of this dataset
for row in lenghtstay_data.itertuples():
    unique_selector = (messages["pid"]==row.pid) & (messages["study_wk"]== row.study_wk)
    edd_wk.append(messages.loc[unique_selector, 'edd_wk'].iloc[0])
    participant_msgs.append(len(messages[(messages["sent_by"]=="participant") & unique_selector])) 
    nurse_msgs.append(len(messages[(messages["sent_by"]=="nurse") & unique_selector]) )
    system_msgs.append(len(messages[(messages["sent_by"]=="system") & unique_selector]) )

lenghtstay_data['edd_wk'] = edd_wk 
lenghtstay_data['participant_msgs'] = participant_msgs 
lenghtstay_data['nurse_msgs'] = nurse_msgs 
lenghtstay_data['system_msgs'] = system_msgs 

#write length of stay summary
lenghtstay_data.to_csv("outputs/data/length_of_stay.csv")

#participant characteristics vs messaging
    #pid
    #study_wk
    #participant_msgs
    #nurse_msgs
    #system_msgs
    #total_msgs
    #related_msgs
    #received_within_3days
    #msg_chars_x
edd_wk = list()
participant_msgs = list()
nurse_msgs = list()
system_msgs = list()
related_response = list()
received_within_3days =list()

part_data = messages.groupby(['pid','study_wk']).size().reset_index().rename(columns={0:'total_msgs'}) #getting all study_wk - study_wk is the primary key of this dataset
for row in part_data.itertuples():
    unique_selector = (messages["pid"]==row.pid) & (messages["study_wk"]== row.study_wk)
    edd_wk.append(messages.loc[unique_selector,'edd_wk'].iloc[0])
    participant_msgs.append(len(messages[(messages["sent_by"]=="participant") & unique_selector])) 
    nurse_msgs.append(len(messages[(messages["sent_by"]=="nurse") & unique_selector]) )
    system_msgs.append(len(messages[(messages["sent_by"]=="system") & unique_selector]) )
    
    #check the response messages if we got related responses to the system message.
    response=(messages["related"]==True)
    related_response.append(len(messages[response & (messages["sent_by"]=="participant") & unique_selector]))

part_data['edd_wk'] = edd_wk 
part_data['participant_msgs'] = participant_msgs 
part_data['nurse_msgs'] = nurse_msgs 
part_data['system_msgs'] = system_msgs 
part_data['related_response'] = related_response    

#To DO
#received_within_3days
#msg_chars_x

#write length of stay summary
part_data.to_csv("outputs/data/part_xteristics.csv")

#msgsData = messages[['pid','timestamp','external','delta_human','delta','study_wk','edd_wk','sent_by','original','has_response']]

