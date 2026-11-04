"""
Amelia Stalter 151193060
YEB.440 Voimalaitokset ja kytketyt energiajärjestelmät
Tietokoneharkka 2: Sähkön hinnan ennustus neuroverkolla
"""

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ===========================
# 1. LUE DATA
# ===========================

train_df = pd.read_excel("Sähkönhintaennustus2020.xlsx")
test_df = pd.read_excel("Sähkönhintaennustus2021.xlsx")

# Muutetaan mahdolliset '-' arvot puuttuviksi
train_df.replace("-", np.nan, inplace=True)
test_df.replace("-", np.nan, inplace=True)

# Muutetaan kaikki numeerisiksi
for col in train_df.columns:
    train_df[col] = pd.to_numeric(train_df[col], errors="coerce")

for col in test_df.columns:
    test_df[col] = pd.to_numeric(test_df[col], errors="coerce")

# Täytetään puuttuvat arvot sarakkeen keskiarvolla
train_df.fillna(train_df.mean(), inplace=True)
test_df.fillna(test_df.mean(), inplace=True)

# ===========================
# 2. MÄÄRITÄ PIIRTEET JA TARGET
# ===========================

# Viimeinen sarake = sähkönhinta
target_col = train_df.columns[-1]

X_train = train_df.drop(columns=[target_col])
y_train = train_df[target_col]

X_test = test_df.drop(columns=[target_col])
y_test = test_df[target_col]

# ===========================
# 3. SKAALAUS
# ===========================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===========================
# 4. NEUROVERKKO
# ===========================

model = Sequential([
    Dense(128, activation="relu",
          input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.2),

    Dense(64, activation="relu"),
    Dropout(0.2),

    Dense(32, activation="relu"),

    Dense(1)
])

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=20,
    restore_best_weights=True
)

history = model.fit(
    X_train_scaled,
    y_train,
    validation_split=0.2,
    epochs=500,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# ===========================
# 5. ENNUSTEET
# ===========================

y_pred = model.predict(X_test_scaled).flatten()

# ===========================
# 6. TARKKUUSMITTARIT
# ===========================

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"MAE  = {mae:.2f}")
print(f"RMSE = {rmse:.2f}")
print(f"R²   = {r2:.4f}")

# ===========================
# 7. TULOSTA ENSIMMÄISET
# ===========================

results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

print(results.head(20))

# ===========================
# 8. KUVAA TULOKSET
# ===========================

import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))
plt.plot(y_test.values[:500], label="Todellinen")
plt.plot(y_pred[:500], label="Ennuste")
plt.xlabel("Aika")
plt.ylabel("Sähkön hinta")
plt.legend()
plt.grid()
plt.show()
