# expected probable Stats (xpStats)
xpStats uses SciKit Learn Neural Network to estimate batting average on baseballs hit using launch speed and launch angle. It is similar to xBA and xSLG provided by Baseball Savant/Statcast but uses a neural network instead of KNN regression.

xpStats is to look up expected probable batting average and slugging percentage per batted ball in play, user must enter launch speed and launch angle. Must have SciKit Learn (sklearn) library installed.

Expect the model to take around 5 minutes to train. But, once it is trained there is no need to retrain it while the script is running. If you want to visualize the result, visualizations take a couple minutes to create.
