import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("trading_data.csv", sep=';')
resin_df = df[df['product'] == 'KELP'].copy()

# Force numeric type
price_cols = [
    'bid_price_1', 'bid_price_2', 'bid_price_3',
    'ask_price_1', 'ask_price_2', 'ask_price_3',
    'mid_price'
]
for col in price_cols:
    resin_df[col] = pd.to_numeric(resin_df[col], errors='coerce')

# Drop rows without mid_price (optional)
resin_df.dropna(subset=['mid_price'], inplace=True)

# Plot
plt.figure(figsize=(14, 6))
plt.plot(resin_df['timestamp'], resin_df['bid_price_1'], label='Bid 1', color='lime')
plt.plot(resin_df['timestamp'], resin_df['bid_price_2'], label='Bid 2', color='green')
plt.plot(resin_df['timestamp'], resin_df['bid_price_3'], label='Bid 3', color='darkgreen')

plt.plot(resin_df['timestamp'], resin_df['ask_price_1'], label='Ask 1', color='red')
plt.plot(resin_df['timestamp'], resin_df['ask_price_2'], label='Ask 2', color='orangered')
plt.plot(resin_df['timestamp'], resin_df['ask_price_3'], label='Ask 3', color='darkred')

plt.plot(resin_df['timestamp'], resin_df['mid_price'], label='Mid price', linestyle='--', color='gray')

plt.title("KELP - Price")
plt.xlabel("Timestamp")
plt.ylabel("Price")
plt.ylim(2020, 2040)  # Force correct y-scale
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
