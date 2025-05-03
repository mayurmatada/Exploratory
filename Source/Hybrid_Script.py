# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# %% Load and clean data
data = pd.read_csv(r"C:\Users\Mayur\Documents\College\4th sem\Exploratory\Data\Sentinel1_MODIS_SM_Masked_Urban_YellowRiver_11km.csv")


def clean_data(data):
    data = data.dropna(subset=['LAI', 'SoilMoisture', 'VH', 'VV', 'date'])
    data = data.groupby('date').mean(numeric_only=True).reset_index()
    data = data.drop(columns=['SoilRoughness_placeholder', 'Frequency_GHz'])
    data['Year'] = pd.to_datetime(data['date']).dt.year
    data['Month'] = pd.to_datetime(data['date']).dt.month
    data['Day'] = pd.to_datetime(data['date']).dt.day
    data = data.drop(columns=['date', 'Year'])
    data['Month'] = pd.to_numeric(data['Month'], errors='coerce')
    data['Day'] = pd.to_numeric(data['Day'], errors='coerce')
    data['Month_Sin'] = np.sin(2 * np.pi * (data['Month'] / 12))
    data['Month_Cos'] = np.cos(2 * np.pi * (data['Month'] / 12))
    data['Day_Sin'] = np.sin(2 * np.pi * (data['Day'] / 30))
    data['Day_Cos'] = np.cos(2 * np.pi * (data['Day'] / 30))
    data = data.drop(columns=['Month', 'Day'])

    # Convert VV and VH from dB to linear
    data['VV'] = 10 ** (data['VV'] / 10)
    data['VH'] = 10 ** (data['VH'] / 10)

    scaler_vv_vh = StandardScaler()
    data[['VV', 'VH']] = scaler_vv_vh.fit_transform(data[['VV', 'VH']])

    scaler_sm_lai = StandardScaler()
    data[['SoilMoisture', 'LAI']] = scaler_sm_lai.fit_transform(data[['SoilMoisture', 'LAI']])

    data['IncidenceAngle'] = data['IncidenceAngle'] * 0.1 / data['SoilMoisture'].std()
    return data


data_clean = clean_data(data.copy())

# %%
feature_cols = ['SoilMoisture', 'LAI', 'IncidenceAngle', 'Month_Sin', 'Month_Cos', 'Day_Sin', 'Day_Cos']
target_cols_vv = ['VV']
target_cols_vh = ['VH']

X = data_clean[feature_cols].values
y_vv = data_clean[target_cols_vv].values
y_vh = data_clean[target_cols_vh].values

X_train, X_test, y_vv_train, y_vv_test, y_vh_train, y_vh_test = train_test_split(
    X, y_vv, y_vh, test_size=0.2, random_state=42
)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_vv_train_tensor = torch.tensor(y_vv_train, dtype=torch.float32)
y_vv_test_tensor = torch.tensor(y_vv_test, dtype=torch.float32)
y_vh_train_tensor = torch.tensor(y_vh_train, dtype=torch.float32)
y_vh_test_tensor = torch.tensor(y_vh_test, dtype=torch.float32)

# %% WCM-Inspired Module


class WCMInspiredModule(nn.Module):
    def __init__(self):
        super(WCMInspiredModule, self).__init__()
        self.soil_transform = nn.Sequential(nn.Linear(1, 8), nn.ReLU(), nn.Linear(8, 1))
        self.veg_transform = nn.Sequential(nn.Linear(1, 8), nn.ReLU(), nn.Linear(8, 1))
        self.angle_transform = nn.Sequential(nn.Linear(1, 8), nn.ReLU(), nn.Linear(8, 1))

    def forward(self, SM, LAI, IncAngle):
        soil_sig = self.soil_transform(SM)
        veg_attn = torch.exp(-F.relu(self.veg_transform(LAI)))
        angle_mod = self.angle_transform(IncAngle)
        sigma0 = soil_sig * veg_attn + angle_mod
        return sigma0

# %% Full Hybrid Model


class FullHybridModel(nn.Module):
    def __init__(self):
        super(FullHybridModel, self).__init__()
        self.phys_layer = WCMInspiredModule()
        self.mlp = nn.Sequential(
            nn.Linear(8, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        SM = x[:, 0:1]
        LAI = x[:, 1:2]
        IncAngle = x[:, 2:3]
        temporal = x[:, 3:]

        wcm_out = self.phys_layer(SM, LAI, IncAngle)
        x_cat = torch.cat([wcm_out, x], dim=1)
        return self.mlp(x_cat)

# %% Train model function


def train_model(model, X_train, y_train, X_test, y_test, label):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_output = model(X_test)
                val_r2 = r2_score(y_test.numpy(), val_output.numpy())
            print(f"[{label}] Epoch {epoch}: Train Loss = {loss.item():.4f}, Val R² = {val_r2:.4f}")

    model.eval()
    with torch.no_grad():
        y_pred = model(X_test).numpy()
        r2 = r2_score(y_test.numpy(), y_pred)
    print(f"[{label}] Final R²: {r2:.4f}")
    return model


# %% Train for VV
model_vv = FullHybridModel()
model_vv = train_model(model_vv, X_train_tensor, y_vv_train_tensor, X_test_tensor, y_vv_test_tensor, "VV")

# %% Train for VH
model_vh = FullHybridModel()
model_vh = train_model(model_vh, X_train_tensor, y_vh_train_tensor, X_test_tensor, y_vh_test_tensor, "VH")
