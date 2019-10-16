# expected probable Stats (xpStats)
xpStats uses SciKit Learn's K-Nearest Neighbors to estimate batting average and slugging percentage on baseballs hit using launch speed and launch angle. It is similar to xBA and xSLG provided by Baseball Savant/Statcast but does not use sprint speed like the MLB does as it is not an attribute provided by Baseball Savant.

xpStats is to look up expected probable batting average and slugging percentage per batted ball in play, user must enter launch speed and launch angle. Must have SciKit Learn (sklearn) library installed.

Once it is trained there is no need to retrain it while the script is running. If you want to visualize the result, visualizations take a couple minutes to create.
