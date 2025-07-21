import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from pathlib import Path

path = Path("C:/Users/oxo97/Downloads/CTA_-_Ridership_-_Daily_Boarding_Totals_20250309.csv")
df = pd.read_csv(path, parse_dates=["service_date"])
print(df.info())
df.columns = ["date", "day_type", "bus", "rail", "total"]  # short name
df = df.sort_values("date").set_index("date")
df = df.drop("total", axis=1)  # total delete
df = df.drop_duplicates()  # duplication delete
print(df.head())

# df["2019-03":"2019-05"].plot(grid=True, marker=".", figsize=(8, 3.5))
# plt.show()

diff_7 = df[["bus", "rail"]].diff(7)["2019-03":"2019-05"]
fig, axs = plt.subplots(2, 1, sharex=True, figsize=(8, 5))
axs[0].set_ylim([0, 900000])
df.plot(ax=axs[0], legend=False, marker=".")
df.shift(7).plot(ax=axs[0], grid=True, legend=False, linestyle=":")
diff_7.plot(ax=axs[1], grid=True, marker=".")
plt.show()
print("last day = %s" % (df.index[-1]))

period = slice("2001", "2019")
df_monthly = df.select_dtypes(include=["number"]).resample('ME').mean()
rolling_average_12_months = df_monthly[period].rolling(window=12).mean()

fig, ax = plt.subplots(figsize=(8, 4))
df_monthly[period].plot(ax=ax, marker=".")
rolling_average_12_months.plot(ax=ax, grid=True, legend=False)
plt.show()
df_monthly[period].diff(12).plot(grid=True, marker=".", figsize=(8, 3))
plt.show()


def to_windows(dataset, size):
    dataset = dataset.window(size, shift=1, drop_remainder=True)
    return dataset.flat_map(lambda window_ds: window_ds.batch(size))


my_series = tf.data.Dataset.range(7)
dataset = to_windows(to_windows(my_series, 3), 4)
dataset = dataset.map(lambda S: (S[:, 0], S[:, 1:]))
print(list(dataset))


dataset = to_windows(tf.data.Dataset.range(6  ), 4)
dataset = dataset.map(lambda window: (window[:-1], window[-1]))
print(list(dataset.batch(2)))

model = tf.keras.Sequential()[

    tf.keras.layers.SimpleRNN(30,input_shape=(30,1)),
    tf.keras.layers.Dense(1)
]

model.compile(loss="mse",optimizer="Nadam", metrics=["accuracy"])
model.fit()

