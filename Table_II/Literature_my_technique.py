QoE1=1.25
QoE2=1.5 #QoE1+QoE1*5/100

alpha=gamma=1
beta=4.3
lenght=193 #seconds
kbps=[0.3, 0.75, 1.2, 1.85, 2.85, 4.3] #Mbps
x3=0
chunks=48
x1=kbps[-1]*chunks
#QoE1
#QoE1=alpha*x1-beta*x21-gamma*x3
x21=(alpha*x1-QoE1+gamma*x3)/beta
#QoE2
#QoE2=alpha*x1-beta*x22-gamma*x3
x22=(alpha*x1-QoE2+gamma*x3)/beta

#x21 and x22 are in milliseconds

#comyco
QoE1=67
QoE2=70 #QoE1+QoE1*5/100

alpha=0.8469
gamma=0.2979
beta=28.7959
delta=1.0610
lenght=193 #seconds
chunks=48
#vmaf=different value for each chunks based on bitrate
vmafmax=100
x3=x4=0
#x1 is vmaf now
x1=vmafmax*chunks
#QoE1
#QoE1=alpha*x1-beta*x21+gamma*x3-theta*x4
x21=(alpha*x1-QoE1+gamma*x3-delta*x4)/beta
#QoE2
#QoE2=alpha*x1-beta*x22-gamma*x3
x22=(alpha*x1-QoE2+gamma*x3-delta*x4)/beta