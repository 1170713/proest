# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 12:49:47 2021

@author: Sofia
"""


import os.path as path
import numpy as np
from scipy.fft import fft,fftfreq
import librosa
import matplotlib.pyplot as plt
from praatio import tgio
import math


dir_path = r"C:\Users\Sofia\Desktop\PROEST\som"
person_number = "10"


tier_name = 'sentence - phones'
    
def get_spectrum(fname,nfft):    
    x, fs = librosa.load(fname)
    
    y = np.zeros((2,2))
    return y

def get_duration(phoneme, text_grid):
    print('Processing /',phoneme,'/...')
    tgn = text_grid.tierDict['sentence - phones'].find(phoneme)
    dur = np.zeros(len(tgn))
    for idx,k in enumerate(tgn):
        dur[idx] = text_grid.tierDict[tier_name].entryList[k].end 
        - text_grid.tierDict[tier_name].entryList[k].start
        print('FREE ({}): Mean Duration (std) (ms): {}({})'.format(idx+1,np.round(np.mean(dur)*1000,1)
                                                                   ,np.round(np.std(dur)*1000,1)))
    

def get_durations(phoneme, tg1, tg2):
    print('Processing /',phoneme,'/...')
    get_duration(phoneme, tg1)
    get_duration(phoneme, tg2)

def total_durations(tg1, tg2):
    t_start = tg1.tierDict[tier_name].entryList[0].start
    t_end = tg1.tierDict[tier_name].entryList[-1].end
    d1 = t_end - t_start
    t_start = tg2.tierDict[tier_name].entryList[0].start
    t_end = tg2.tierDict[tier_name].entryList[-1].end
    d2 = t_end - t_start
    print('Duracao total: ',np.round(d1,1),'-',np.round(d2,1))
    
#%% SPECTRUM
def process_data(file_path, phoneme, textgrid):
    tgn = textgrid.tierDict['sentence - phones'].find(phoneme)
    x, fs = librosa.load(file_path)
    nfft=1024
    S=np.zeros((len(tgn),nfft//2))
    for idx,k in enumerate(tgn):
        sini = int(textgrid.tierDict[tier_name].entryList[k].start*fs)
        send = int(textgrid.tierDict[tier_name].entryList[k].end*fs)
        s = x[sini:send]
        temp = fft(s, nfft)
        S[idx,:] = 2/nfft*np.abs(temp[0:nfft//2])
    w=fftfreq(nfft,1/fs)[:nfft//2]
    S=S/abs(S.max())
    S=20*np.log10(S)
    return (w,S)

   


def create_plot( phoneme, file_path1, file_path2 ,textgrid1, textgrid2, difference, diff_start, diff_end):
    
    (w, S1) = process_data(file_path1,phoneme,tg_free)
    (w, S2) = process_data(file_path2,phoneme,tg_nasal)

    if difference:
        newS1= np.mean(S1,0)
        newS2= np.mean(S2,0)
        dif = abs(newS2-newS1)
        diftotal = math.sqrt(sum(dif[diff_start:diff_end]**2))
        print(diftotal)
        plt.plot(w[diff_start:diff_end],dif[diff_start:diff_end], color=(0,1,0))
        plt.title('Difference- Subject {}- Phoneme {}'.format(person_number,phoneme))
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid()
        plt.text(6000, -9, "Total difference = {:0.2f}".format(diftotal), size=10,
         ha="center", va="center",
         bbox=dict(boxstyle="round",
                   ec=(1., 0.5, 0.5),
                   fc=(0, 0.8, 0.8),
                   )
         )
        plt.show()
        
    else: 
        plt.plot(w,np.mean(S1,0), label="normal") 
        plt.plot(w,np.mean(S2,0),'-r', label="obstruction" )
        plt.fill_between(w, np.min(S1,0), np.max(S1,0), alpha=0.2, color=None)
        plt.fill_between(w, np.min(S2,0), np.max(S2,0), alpha=0.2, color= (1,0,0))
        plt.title('Normal vs Obstruction - Subject {} - Phoneme {}'.format(person_number,phoneme))
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid()
        plt.show()


# Main
free_wav = path.join(dir_path, "{}_free.wav".format(person_number) )
nasal_wav = path.join(dir_path, "{}_nasal.wav".format(person_number) )
tg_free = tgio.openTextgrid(path.join(path.join(dir_path, "DARLA")
                                      , "{}_free_DARLA.TextGrid".format(person_number) ))
tg_nasal = tgio.openTextgrid(path.join(path.join(dir_path, "DARLA")
                                       , "{}_nasal_DARLA.TextGrid".format(person_number) ))
    

#%% Durations
total_durations(tg_free, tg_nasal)
get_durations('M', tg_free, tg_nasal)
get_durations('N', tg_free, tg_nasal)
get_durations('NG', tg_free, tg_nasal)    

create_plot("M", 
            free_wav,
            nasal_wav,
            tg_free,
            tg_nasal,
            None,
            None,
            None)
