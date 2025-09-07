####################################### OFDM PROCESS ##########################
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib as plt


#prepare empty arrays
recieved_frame=np.zeros(80,dtype=complex) 
transmit_frame=np.zeros(80,dtype=complex)


######################################## qam256 modulating and transmition function ############################
 
def qam_256_modulator_transmitor(frame: np.ndarray) -> np.ndarray:
    reserved_spots=[7,21,57,43,0,31,30,29,28,27,26,32,33,34,35,36] 
    s=0
    modulated_row = []
    for q in range(0,64):
        if q not in reserved_spots:    
            num = int(frame[s])
            I = (((num >> 4) & 0xF) * 2) - 15
            Q = ((num & 0xF) * 2) - 15
            modulated_row.append(complex(I, Q))
            s+=1
        elif q in (21,7,57, 43):
            modulated_row.append(-1)  #  add a pilot
        else:
            modulated_row.append(0)  # add a gaurd
    
    qam_256_frame_time_1=np.fft.ifft(modulated_row)
    ## adding cyclic prefix 16 last of the 48 
    transmit_frame[0:16]=qam_256_frame_time_1[48:]
    transmit_frame[16:]=qam_256_frame_time_1
    return transmit_frame


###########################channel################################

def channel_multipath(tansmition_frame:np.ndarray)->np.ndarray:
    h =[1, 0.8, 0.5, 0.2] #channel multipath
    multipath_signal_1=np.convolve(tansmition_frame,h)
    ###chanell noise
    SNR_db=20
    data_power=np.mean(np.abs(tansmition_frame)**2)
    noise_power=data_power/ (10**(SNR_db/10))
    noise = np.sqrt(noise_power/2) * (np.random.randn(*multipath_signal_1.shape) + 1j*np.random.randn(*multipath_signal_1.shape))
    #print("noise sample:", noise[:3])  
    return multipath_signal_1+ noise

    
    
###################### reciver rx ########################
def reciver(multipath_sig:np.ndarray)->np.ndarray:
    h =[1, 0.8, 0.5, 0.2] #channel multipath KNOWN BY RECEIVER
    N=64
    H=np.fft.fft(h,N)
    reserved_spots=[7,21,57,43,0,31,30,29,28,27,26,32,33,34,35,36]
    recived_data=np.zeros(48,dtype= int)
    recived_multipath_sig=  multipath_sig 
    recived_sig_no_prefix= recived_multipath_sig[16:80] # clip the CP and the extra samples form convolution
    symb=np.fft.fft(recived_sig_no_prefix) 
    symb_estimated= symb/H    ### estimate the input by devision of chanell response
    #print(symb_estimated)
    s=0
    for i in range (64):
        if i  not in reserved_spots:
            I=int(np.round(symb_estimated[i].real+15)//2)
            Q=int(np.round(symb_estimated[i].imag+15)//2)
            #clip to values [0:15]
            I= min(max(I,0),15)
            Q= min(max(Q,0),15)
            R=(I<<4)|Q
            recived_data[s]=R
            s+=1
    #print(recived_data)        
    return recived_data        

   
#################### main prog ###################
# open data file and read the date 

with open("C:\\Users\\ruppi\\Desktop\\python\\OFDM_PROJ\\bits_10000.txt","r") as data_file :
    #get the data bits
    bits=np.array([int(line.strip())for line in data_file], dtype=np.uint)
    # 48 data subcarriers *8 bit per simbol = 384 data bits per frame
    size_data=bits.size
    modulu_data= size_data%384
    padding=384 - modulu_data
    if(not( size_data%384 ==0)):
        # if the data  isnt full mult of 384 - zero padd
        padded_data = np.pad(bits, (0, padding), constant_values=0) 

    symbols = padded_data.reshape(-1,8)
    powers= 2**np.arange(7,-1,-1) # powers = [ 128,64,32,16,8,4,2,1]
    parallel= symbols @ powers # convert each 8 bits to decimal value->symbol in a matrix
    parallel_matrix=parallel.reshape(-1,48) # each row (27 rows) in this matrix is a frame with 48 symbols
    num_of_frames= parallel_matrix.shape[0]
    # prepare an empty matrix to recieve the data ->27 frames of 48 data symbols
    recieved_data_matrix=np.zeros((27,48),dtype=int)
    

for i in range (num_of_frames):
    # send each row to modulation and transmition
    transmit_frame=qam_256_modulator_transmitor(parallel_matrix[i,:]) 
    
    # transmit frame through channel with multipath and awgn
    multipath_signal=channel_multipath(transmit_frame)
    
    # recieve the data (removing cp and demodulation) 
    recieved_data_matrix[i,:]=reciver(multipath_signal)

#print(parallel_matrix)
#print(recieved_data_matrix)
print(np.max(parallel_matrix-recieved_data_matrix))

