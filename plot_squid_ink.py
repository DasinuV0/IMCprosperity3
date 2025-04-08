import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("trading1.csv", sep=';')
squid_ink_df = df[df['product'] == 'SQUID_INK'].copy()

# Force numeric type
price_cols = [
    'bid_price_1', 'bid_price_2', 'bid_price_3',
    'ask_price_1', 'ask_price_2', 'ask_price_3',
    'mid_price'
]

for col in price_cols:
    squid_ink_df[col] = pd.to_numeric(squid_ink_df[col], errors='coerce')

# Drop rows without mid_price (optional)
squid_ink_df.dropna(subset=['mid_price'], inplace=True)

# Plot
plt.figure(figsize=(14, 6))
plt.plot(squid_ink_df['timestamp'], squid_ink_df['bid_price_1'], label='Bid 1', color='lime')
plt.plot(squid_ink_df['timestamp'], squid_ink_df['bid_price_2'], label='Bid 2', color='green')
plt.plot(squid_ink_df['timestamp'], squid_ink_df['bid_price_3'], label='Bid 3', color='darkgreen')

plt.plot(squid_ink_df['timestamp'], squid_ink_df['ask_price_1'], label='Ask 1', color='red')
plt.plot(squid_ink_df['timestamp'], squid_ink_df['ask_price_2'], label='Ask 2', color='orangered')
plt.plot(squid_ink_df['timestamp'], squid_ink_df['ask_price_3'], label='Ask 3', color='darkred')

plt.plot(squid_ink_df['timestamp'], squid_ink_df['mid_price'], label='Mid Price', color='black')

plt.title('Squid Ink Price Levels')
plt.xlabel('Timestamp')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()







