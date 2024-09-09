import obspy
from obspy import read
from obspy import read_inventory
from obspy.imaging.beachball import beachball
from obspy.clients.fdsn import Client
from obspy import UTCDateTime


client= Client("IRIS")
t1 = UTCDateTime("2020-01-02T18:23:53.011303Z")
t2=t1+300 #5 dakika
st1= client.get_waveforms("G", "INU", "00", "BHZ", t1, t2)

st1.plot(tyoe="relative",method="full",picker=True)

st1.filter('bandpass', freqmin=0.1, freqmax=1.0)

st1.plot()

from obspy.signal.trigger import classic_sta_lta, trigger_onset
tr = st1[0]  
cft = classic_sta_lta(tr.data, int(5 * tr.stats.sampling_rate), int(10 * tr.stats.sampling_rate))
onset = trigger_onset(cft, 1.5, 0.5)

p_pick = UTCDateTime("2020-01-02T18:24:45")  # aproximate P-wave arrival time of japan earhquake

from obspy.core.event import Pick, WaveformStreamID
pick = Pick(time=p_pick, waveform_id=WaveformStreamID(network_code="IU", station_code="ANMO"),
            phase_hint="P", onset="impulsive", polarity="positive")  # "positive" for upward (?)
