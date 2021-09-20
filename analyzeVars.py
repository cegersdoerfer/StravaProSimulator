import matplotlib.pyplot as plt
import pandas as pd
from proSimulator import ProDataSimulator
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import math



activities_df = pd.read_csv("/Users/chris_egersdoerfer/Desktop/proData-csv/test_all_male")

scale= StandardScaler()

fig = plt.figure()

ax1 = fig.add_subplot(2,5,1)
ax1.scatter(activities_df['e_gain'], activities_df['time']/60)
#ax1.scatter(activities_df['e_gain'], activities_df['avgSpeed']**2)


ax2 = fig.add_subplot(2,5,2)
ax2.scatter(activities_df['e_loss'], activities_df['time']/60)
#ax2.scatter(activities_df['e_loss'], activities_df['avgSpeed']**2)


ax3 = fig.add_subplot(2,5,3)
ax3.scatter(activities_df['distance'], activities_df['time']/60)
#ax3.scatter(activities_df['distance'], activities_df['avgSpeed']**2)


ax4 = fig.add_subplot(2,5,4)
ax4.scatter(activities_df['avgSpeed'], activities_df['time']/60)
#ax4.scatter(activities_df['avgSpeed'], activities_df['avgSpeed']**2)


#scaled_data = scale.fit_transform(activities_df[['turns']]) 
ax5 = fig.add_subplot(2,5,5)
ax5.scatter(activities_df['turns']**(1/2), activities_df['time']/60)
#ax5.scatter(activities_df['turns'], activities_df['avgSpeed']**2)

maxDownHill = max(activities_df.loc[:, 'downHillTurns'])
minDownHill = min(activities_df.loc[:, 'downHillTurns'])
dif = maxDownHill - minDownHill
ax6 = fig.add_subplot(2,5,6)
ax6.scatter(((activities_df['downHillTurns']/dif) * activities_df['distance'])**(1/4), activities_df['time']/60)
#ax6.scatter(activities_df['avgTurnDegree'], activities_df['avgSpeed']**2)


ax7 = fig.add_subplot(2,5,7)
ax7.scatter(activities_df['avgTurnLength'] * activities_df['turns'], activities_df['time']/60)
#ax7.scatter(activities_df['avgTurnLength'], activities_df['avgSpeed']**2)

ax8 = fig.add_subplot(2,5,8)
ax8.scatter(activities_df['turns']/activities_df['distance'], activities_df['time']/60)

ax9 = fig.add_subplot(2,5,9)
ax9.scatter(activities_df['turns']/activities_df['e_gain'], activities_df['time']/60)

ax10 = fig.add_subplot(2,5,10)
ax10.scatter(activities_df['e_gain'] - activities_df['e_loss'], activities_df['time']/60)

plt.show()





