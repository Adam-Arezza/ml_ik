from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import RegressorChain
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import numpy as np


parsed_data = []
with open("pose_data.txt", "r") as data_file:
    for line in data_file.readlines():
        line = line.strip()
        data_line = []
        line = line.split(",")
        for val in line:
            if val != '':
                data_line.append(float(val))
        parsed_data.append(data_line)
    data_file.close()

parsed_data.pop(-1)
data = np.array(parsed_data)
x = data[:,0:7].astype(np.float32)
y = data[:,7:].astype(np.float32)
y[:, 2:] = y[:, 2:] + 90
print(data[0:1])
print(x[0:1])
print(y[0:1])

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.15, random_state=25)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#print(x[0:5])
#print(y[0:5])

#svr model
model = SVR()
multi_regressor = MultiOutputRegressor(model)
multi_regressor.fit(x_train, y_train)
preds = multi_regressor.predict(x_test)
preds = np.round(preds, decimals=0)
score = mean_absolute_error(y_test, preds)
print("-----------------------SVR Results-----------------------")
print(preds[:5])
print(y_test[:5])
print(f"mean absolute error: {score}")
print("\n")


#kn model
kn_model = KNeighborsRegressor(n_neighbors=7)
kn_model.fit(x_train, y_train)
kn_preds = kn_model.predict(x_test)
kn_preds = np.round(kn_preds, decimals=0)
kn_score = mean_absolute_error(y_test, kn_preds)

print("-----------------------KN Results-----------------------")
print(kn_preds[:5])
print(y_test[:5])
print(f"mean absolute error: {kn_score}")
print("\n")

#rf model
rf_model = RandomForestRegressor(n_estimators=500, max_depth=10)
rf_model.fit(x_train, y_train)
rf_preds = rf_model.predict(x_test)
rf_preds = np.round(rf_preds, decimals=0)
rf_score = mean_absolute_error(y_test, rf_preds)

print("-----------------------rf Results-----------------------")
print(rf_preds[:5])
print(y_test[:5])
print(f"mean absolute error: {rf_score}")
print("\n")


#svr = SVR()
#r_chain = RegressorChain(base_estimator=svr)
#r_chain.fit(x_train,y_train)
#r_chain_preds = r_chain.predict(x_test)
#r_chain_preds = np.round(r_chain_preds, decimals=0)
#r_chain_score = mean_absolute_error(y_test, r_chain_preds)
#
#print("-----------------------regressor chain svr Results-----------------------")
#print(r_chain_preds[:5])
#print(y_test[:5])
#print(f"mean absolute error: {r_chain_score}")
#print("\n")
#


