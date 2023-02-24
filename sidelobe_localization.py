#! /usr/bin/env python

import click
from glob import glob
import numpy as np
import os, shutil
import subprocess
import json
import matplotlib.pyplot as plt
import operator
import datetime, time
from astropy.coordinates import FK5
from astropy import units as u
from astropy.coordinates import SkyCoord
from scipy.signal import find_peaks
from scipy import stats
import math
import matplotlib.ticker as ticker

from iautils.conversion import chime_intensity
from frb_common import common_utils
from frb_common import configuration_manager
from iautils import spectra, spectra_utils, cascade
from scipy.spatial.transform import Rotation
import chime_frb_constants as constants

np.set_printoptions(precision=4)

MASK = os.path.join('../data/bad_channel_16K.mask')
MOUNT_POINT = '/data/frb-archiver/'
FRB_CONFIGS = os.path.join('/home/zpleunis/software/frb-configs/')

inst_lat = 49.3203
inst_long = -119.6175
R2D = 180./np.pi
D2R = np.pi/180.
lat = inst_lat*D2R
rot = Rotation.from_euler("x", -constants.TELESCOPE_ROTATION_ANGLE, degrees=True) #telescope rotation

def get_LST_ZA(UTCtime, beamID):
    """
    get overhead meridian RA (LST) from UTC time now
    get Zenith angle from beam ID and beam spacking look-up array.
    """
    N_feeds = 256
    d = 0.3048
    c = 3.e8
    Ref_freq = 1./ (np.sin(60*D2R) / c / (N_feeds/2) * d * N_feeds *1.e6)
    Beam_Reference=[] #-60 to 60 the zenith angle
    for m in range(N_feeds):
        Beam_Reference.append(np.arcsin( c * (m-N_feeds/2) / (Ref_freq*1.e6) / N_feeds / d)  * R2D)
    year = UTCtime.year
    month = UTCtime.month
    if (month < 3):
        month = month + 12
        year = year - 1
    day = UTCtime.day
    hour = UTCtime.hour
    minute = UTCtime.minute
    second  = UTCtime.second
    JD = 2-int(year/100.)+int(int(year/100.)/4.)+int(365.25*year)+int(30.6001*(month+1))+day+1720994.5
    T= (JD-2451545.0)/36525.0
    T0 = ((6.697374558 + (2400.051336*T) + (0.000025862*T*T) + 24*99999999999)  )%24
    UT = hour + minute/60. + second/3600.
    GST = (T0 + UT*1.002737909) % 24.
    LST = GST + inst_long/15.
    while LST < 0:
       LST = LST + 24
    LST = LST%24

    return LST*15, Beam_Reference[beamID]

def _cart2sph(x, y, z):
    """
    Converts cartesian coordinates to spherical coordinates.
    Parameters
    ----------
    x, y, z: float
        coordinates on unit sphere
    Returns
    -------
    phi, theta: float
        polar and azimuthal angles
    """
    phi = np.arccos(z)
    theta = np.arctan2(y, x)
    return phi, theta

def _sph2cart(phi, theta):
    """
    Converts spherical coordinates to cartesian coordinates.
    Parameters
    ----------
    phi: float
        polar angle in [0, pi)
    theta: float
        azimuthal angle in [0, 2*pi)
    Returns
    -------
    x, y, z: float
        Cartesian coordinates
    """
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)
    return x, y, z

def Offset2RADec(x,y, LST, UTCtime):
    phi = np.pi / 2. - np.deg2rad(y)  # polar angle
    theta = np.deg2rad(x) / np.sin(phi)  # azimuthal angle
    # correct for telescope rotation
    # note: frame has 'z' axis at North horizon, and 'x' axis at zenith    phi, theta = _cart2sph(*rot.apply(np.array(_sph2cart(phi, theta)).T).T)
    phi, theta = _cart2sph(*rot.apply(np.array(_sph2cart(phi, theta)).T).T)
    dec = np.arcsin(np.cos(lat) * np.cos(phi) + np.sin(lat) * np.sin(phi) * np.cos(theta))
    cosh = (-1.* (np.cos(phi) - np.cos(lat) * np.sin(dec))/ (np.sin(lat) * np.cos(dec)) )
    sinh = np.sin(phi) * np.sin(theta) / np.cos(dec)
    hour_angle = np.arctan2(sinh, cosh)
    ra_deg = ( LST -  hour_angle*R2D ) %360
    dec_deg = dec*R2D
    #convert to J2000
    yeardec = UTCtime.year + UTCtime.month/12.
    c_fk5 = SkyCoord(ra=float(ra_deg)*u.degree, dec=float(dec_deg)*u.degree, frame='fk5',equinox='J'+str(yeardec))
    c_fk5_2000 = c_fk5.transform_to(FK5(equinox='J2000'))
    RAloc = c_fk5_2000.ra.degree
    DECloc = c_fk5_2000.dec.degree
    return RAloc, DECloc    

def PlotBeamArc(ZA,LST, UTCtime, Offset):
    year = UTCtime.year
    month = UTCtime.month

    xs = np.linspace(-Offset*1.1,Offset*1.1,10)
    xs = np.sort(np.append(xs,0))  #an array of EW offset, including 0 deg (no offset)
    ID =  [i for i, e in enumerate(xs) if round(e,2) == 0.00] #ID for peak point (offset=0)

    RAs = np.zeros((len(xs)))
    DECs = np.zeros((len(xs))) 
    
    for xid in range(len(xs)):
        x = xs[xid]
        RAs[xid], DECs[xid] = Offset2RADec(x,ZA, LST, UTCtime) 

        #Annotating the beam arm plot (ax5) with the EW offset at each point 
        if xid ==  len(xs)-2:
            ax4.annotate("EW="+str(np.round(xs[xid],1))+u'\u00B0',xy=(RAs[xid],DECs[xid]),va='center',color='darkgoldenrod')
        elif xid == ID[0]:
            ax4.text(RAs[xid],DECs[xid-2],'['+str(round(RAs[xid],1))+u'\u00B0'+","+str(round(DECs[xid],1))+u'\u00B0'+"]",fontsize=15,ha='center',va='top',color='darkgoldenrod')
            ax4.scatter(RAs[xid],DECs[xid],marker='o',c='darkgoldenrod',zorder=0,label='nominal')
        elif xid !=0 and xid != len(xs)-1:
            ax4.annotate(" "+str(np.round(xs[xid],1))+u'\u00B0',xy=(RAs[xid],DECs[xid]),va='center',color='darkgoldenrod')

    ax4.plot(RAs,DECs,marker='o',c='darkgoldenrod')
    PeakRA = RAs[ID]
    PeakDec = DECs[ID]
    print("Peak RA, Dec (J2000) in the beam arc plot for beam 1xxx: ", PeakRA, PeakDec)
    return PeakRA, PeakDec


def PlotObserved(x, ZA, LST, UTCtime,PeakRA, PeakDec,MaxOff,MinOff):
    RAmax, DECmax = Offset2RADec(MaxOff,ZA,LST, UTCtime)
    #print("Max off", MaxOff, "RA=", RAmax, "DEC=", DECmax )
    RAmin, DECmin = Offset2RADec(MinOff,ZA,LST, UTCtime)
    #print("Min off", MinOff, "RA=", RAmin, "DEC=", DECmin)
    RAloc, DECloc = Offset2RADec(x,ZA,LST, UTCtime)
    ax4.scatter(RAloc,DECloc,c='k',label='corrected',zorder=10)
    #Uncertainty
    ax4.plot(( RAmax,RAloc),( DECmax,DECloc),c='k',marker='_',zorder=10)
    ax4.plot(( RAmin,RAloc),( DECmin,DECloc),c='k',marker='_',zorder=10)
    RAdiff = np.round(np.max([np.abs(RAmax-RAloc), np.abs(RAmin-RAloc)]),1)
    DECdiff = np.round(np.max([np.abs(DECmax-DECloc), np.abs(DECmin-DECloc)]),1)
    textcoord = '['+str(round(RAloc,1))+"$\pm$"+str(RAdiff)+u'\u00B0'+','+str(round(DECloc,1))+"$\pm$"+str(DECdiff)+u'\u00B0'+']'
    if x>0:
        ax4.text(RAloc,DECloc,textcoord,ha='left',va='top',color='k',fontsize=15)
        ax4.legend(loc=4,fontsize=9)
    else:
        ax4.text(RAloc,DECloc,textcoord,ha='right',va='top',color='k',fontsize=15)
        ax4.legend(loc=3,fontsize=9)
    print("Corrected coordinates (J2000): RA=", RAloc, "Dec=", DECloc)
    print("    Uncertainty: RA=", RAdiff, "Dec=", DECdiff)

def get_frame0_nano(event_time):
    """Get frame0_nano from the frb-configs GitHub repository."""
    commit_hash = configuration_manager.get_config_commit_hash(FRB_CONFIGS,
        str(event_time.split(".")[0]))
    
    cmd = "git --git-dir={}.git show {}:".format(FRB_CONFIGS, commit_hash)
    cmd += "RunParameters/ch_master_curl.json"

    ch_master_curl = subprocess.check_output(cmd, shell=True).decode(
        'string-escape').strip('"')
    frame0_nano = json.loads(ch_master_curl)["frame0_nano"]

    return frame0_nano

def retrieve_parameters(eventid):
    from chime_frb_api.backends import frb_master
    master = frb_master.FRBMaster()
    event = master.events.get_event(eventid)
    mp= event["measured_parameters"][0]
    beam_number = event["beam_numbers"]

    datestr = str(mp["datetime"]).split(' U')[0]
    print(datestr, mp["datetime"])
    date = str(mp["datetime"]).split(' ')[0]
    p = os.path.join(*date.split('-'))
    date = datetime.datetime.strptime(datestr, '%Y-%m-%d %H:%M:%S.%f')

    return beam_number, date, p


def FWHM1(X,Y):
    half_max = (np.max(Y) -np.min(Y)) / 2. + np.min(Y)
    d = np.sign(half_max - np.array(Y[0:-1])) - np.sign(half_max - np.array(Y[1:]))
    left_idx = np.where(d > 0)[0][0]
    right_idx = np.where(d < 0)[0][-1]
    return X[right_idx], X[left_idx] , half_max


@click.command()
@click.option('--eventid', type=int,
    help="Query the database for parameters associated with the event ID.")
@click.option('--t1', type=float, default=2,
    help="Start trial step")
@click.option('--slope', type=int, default=0,
    help="Slope direction")
@click.option('--t2', type=float, default=20,
    help="End trial step")
@click.option('--ts', type=float, default=1,
    help="Start trial step size")
@click.option('--fit', is_flag=True, default=False,
    help="fwhm fit or not")
@click.option('--nsub', default=128,
    help="Number of subbands to reduce the data to.")
@click.option('--outfile', default="",
    help="Output file names of the waterfall plot (.png) and log file (.log).")
@click.option('--dm', default=-1.,
    help="Override DM do not refine database parameters.")
@click.option('--save', is_flag=True, default=False,
    help="Save intensity data to disk.")

def get_intensity(eventid, nsub, outfile, dm, save, t1, t2, ts, slope,fit):

    #find out how many beams
    beam_numbers, UTCtime, path = retrieve_parameters(eventid)
    date = str(UTCtime).split()[0]
    ymd = os.path.join(*date.split('-'))
    print("Event time:", UTCtime, "Path:", path)
    print("Detected beams", beam_numbers)


    #===[TODO] make it only 4 beams somehow
    nbeams = len(beam_numbers)
    PeakSNRs = np.zeros(nbeams)

    #Set up the plot ticks according to input values 
    ax3.xaxis.set_minor_locator(ticker.MultipleLocator(ts))
    ax5.xaxis.set_minor_locator(ticker.MultipleLocator(ts))
    Spacing = np.arange( t1, t2, ts)    
    SumSNR = np.zeros(len(Spacing))

    MHz = np.zeros(4)
    Peaks = np.zeros(4)

    #query database
    for beam in range(nbeams):
        beamid = beam_numbers[beam]

        #Read from 1024-freq npz file
        npzname = '/data/user-data/hhlin/snapshots/'+str(eventid)+'/'+str(eventid)+'_'+str(beamid)+'_intensity_snapshot.npz'
        print("Working on beam", beam,"/",nbeams)
        print("    File:", npzname)

        data= np.load(npzname, allow_pickle=True)
        Intensity = data['intensity']
        #print("Original intensity data shape", Intensity.shape)

        ntime = Intensity.shape[1]
        Intensity = np.sum(Intensity.reshape(nsub,-1,ntime),axis=1)

        #Find on pulse
        TS = np.sum(Intensity,axis=0)
        index, value = max(enumerate(TS), key=operator.itemgetter(1))
        Intensity = Intensity[:,index-10:index+10]
        #print("Working on intensity in shape", Intensity.shape)

        colid = int(str(beamid).zfill(4)[0])  

        #Plot intensity imshow (showing at most 4 one from each column)
        ax = plt.subplot2grid((3,8), (0,colid*2), rowspan=1, colspan=2)
        im = ax.imshow(Intensity,aspect="auto", origin='lower',interpolation=None)
        ax.set_xlabel('Time bin')
        plt.title("Beam"+str(beamid).zfill(4))
        plt.ylabel(str(nsub)+" freq")

        #Plot spectrum 
        TS = np.sum(Intensity,axis=0)
        index, value = max(enumerate(TS), key=operator.itemgetter(1))
        Spectrum = Intensity[:,index]
        ax = plt.subplot2grid((3,8), (1,colid*2), rowspan=1, colspan=2)
        plt.plot(Spectrum,c=colors[colid])
        ax.set_yticklabels([])
        if colid == 0:
            ax.set_ylabel('Spectrum')
        else:
            ax.set_ylabel('')

        #Find peaks in spectrum
        factor = 1.5
        peaks, _ = find_peaks(Spectrum, height=np.std(Spectrum)*factor)
        for i in peaks:
            ax.scatter(i,Spectrum[i],c='k',s=5,zorder=10)
        ax.axhline(y=np.std(Spectrum)*factor,c='y',ls=':',zorder=0)

        snr = np.zeros(len(Spacing))

        for i in range(len(Spacing)):
            firstbin = peaks[0]%Spacing[i]
            j=0
            while int(firstbin+Spacing[i]*j)<nsub:
                snr[i] = snr[i] + Spectrum[int(firstbin+Spacing[i]*j)]
                j=j+1
        ax3.plot(Spacing,snr,label=str(beamid),c=colors[beam])
        
        index, value = max(enumerate(snr), key=operator.itemgetter(1))
        print("    Beam=",str(beamid).zfill(4),"Column ID=", colid," SNR peak at ID=",index, "bin spacing=",np.round(Spacing[index],3), "equiv. MHz=",np.round(Spacing[index]*400/float(nsub),3))
        PeakSNRs[beam] = snr[index]
        firstbin = peaks[0]%Spacing[index]
        j=0
        while int(firstbin+Spacing[index]*j)<nsub:
            ax.axvline(x=int(firstbin+Spacing[index]*j),c='gainsboro',zorder=0)
            j=j+1
        ax3.scatter(Spacing[index],value,c='k',s=5,zorder=10)
        xmax=nsub
        ax.set_xlim([-1,xmax])
        ax.set_xlabel(str(xmax)+' freq chan')
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(10))
        
        SumSNR = SumSNR + snr

    print("Array of peak SNRs", PeakSNRs)
    BestBeam, BestSNR = max(enumerate(PeakSNRs), key=operator.itemgetter(1))
    NSbeam = int(str(beam_numbers[BestBeam]).zfill(4)[1:4])
    print("Best beam=", beam_numbers[BestBeam],"Best NS beam ID=", NSbeam)
    LST, ZA = get_LST_ZA(UTCtime, NSbeam) 
    print("Zenith angle=", np.round(ZA,3), "deg, Meridian LST=", np.round(LST,3) , "deg")

    ax5.plot(Spacing,SumSNR,c='k',label='Sum beams')
    index, value = max(enumerate(SumSNR), key=operator.itemgetter(1))
    AvgMHz = Spacing[index]*400/float(nsub)
    ax5.scatter(Spacing[index],value,c='k',s=5)
    ax5.set_xlim([np.min(Spacing), np.max(Spacing)])

    x=Spacing
    y=SumSNR

    #Assume some uncertainty by default
    w1 = w2 = 10

    if fit:
        w1, w2, hm = FWHM1(x,y)
        #print("FWHM:",w1,w2,hm)
        ax5.plot((w1,w2),(hm,hm),c='c',label='FWHM as error')
    ax5.legend(loc=0,fontsize=6)
    Peaks = Peaks[np.nonzero(Peaks)]

    #Figure out the direction - NOT WORKING VERY WELL! require manual input
    if slope == 0 :
        if np.all(np.diff([Peaks]) > 1)  :  #Increasing -> +ve slope -> +ve EW
            slope = 1
        elif np.all(np.diff([Peaks]) <0) : #Decreasing -> -ve slope -> -ve EW
            slope = -1
    if slope == -1: #Negative EW
        AvgMHz = - np.abs(AvgMHz)
        MHz_max = - w2*400/float(nsub)
        MHz_min = - w1*400/float(nsub)
    if slope == 1: #Postive EW
        AvgMHz = np.abs(AvgMHz)
        MHz_max = w2*400/float(nsub)
        MHz_min = w1*400/float(nsub)

    #Apply empirical replationship 
    OffsetDeg = 772.9433400352848/AvgMHz+1.8440911751394002
    print("Most likely bin spacing=", np.round(Spacing[index],3),", Avg blob separation=", np.round(AvgMHz,3), "MHz, EW offset=", np.round(OffsetDeg,3), "deg")
    MaxOff = 772.9433400352848/MHz_max+1.8440911751394002
    MinOff = 772.9433400352848/MHz_min+1.8440911751394002
    print("   Max EW Offset=", np.round(MaxOff,3),"deg, equiv blob sep=", np.round(MHz_max,3), "MHz")
    print("   Min EW Offset=", np.round(MinOff,3),"deg, equiv blob sep=", np.round(MHz_min,3), "MHz")

    PeakRA, PeakDec = PlotBeamArc( ZA, LST, UTCtime, OffsetDeg) 
    PlotObserved( OffsetDeg , ZA, LST, UTCtime, PeakRA, PeakDec,MaxOff,MinOff)


    #Save plots
    plt.figtext(0.5,0.93,"Event "+str(eventid)+" || using "+str(nbeams)+"("+str(len(beam_numbers))+") beams || Sep.="+str(np.round(AvgMHz,1))+"MHz || EW offset="+str(np.round(OffsetDeg,1))+u'\u00B0',ha="center",fontsize=15)
    plt.savefig("SidelobeAnalysis_Event"+str(eventid)+".png",bbox_inches='tight')
    os.system("chmod 777 SidelobeAnalysis_Event"+str(eventid)+".png")
    os.system("rsync -acvP SidelobeAnalysis_Event"+str(eventid)+".png precommissioning@frb-L4:/home/precommissioning/public_html/sidelobe.png")
    print ("On your browser, visit: https://frb.chimenet.ca/~precommissioning/sidelobe.png")


if __name__ == '__main__':



    #Set up the plot
    fig = plt.figure(figsize=[12,6])
    plt.subplots_adjust(hspace=0.5)
    colors = ['b','r','g','m','tab:blue','tab:red','tab:green','tab:purple','navy','darkred','darkgreen','darkmagenta']

    #per beam snr
    ax3 = plt.subplot2grid((3,8), (2,0), rowspan=1, colspan=2)
    ax3.set_xlabel('Trial gap')
    ax3.grid()
    ax3.set_yticklabels('')

    #sum all beam snr
    ax5 = plt.subplot2grid((3,8), (2,2), rowspan=1, colspan=2)
    ax5.set_xlabel('Trial gap')
    ax5.grid()
    ax5.set_yticklabels('')

    #beam arc plot
    ax4 = plt.subplot2grid((3,8), (2,4), rowspan=1, colspan=4)
    ax4.yaxis.set_label_position("right")
    ax4.yaxis.tick_right()
    ax4.legend(loc=2,fontsize=7)
    ax4.plot([83.63322083],[22.01446111],c='olive',marker='D',label='Crab')
#    ax4.plot([53.2475],[54.57870],c='olive',marker='D',label='B0329')
    ax4.set_xlabel("RA J2000 (deg)")
    ax4.set_ylabel("DEC J2000 (deg)")
    ax4.grid()



    get_intensity()

