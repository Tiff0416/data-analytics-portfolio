import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === 1️⃣ 讀取資料 ===
file_path = 'TXF_R1_1min_data_combined.csv'  # 修改成你的路徑
df = pd.read_csv(file_path, parse_dates=['datetime'])
df.set_index('datetime', inplace=True)

# === 2️⃣ 聚合日線 ===
daily = df.resample('1D').agg({
    'Open': 'first',
    'High': 'max',
    'Low': 'min',
    'Close': 'last',
    'Volume': 'sum'
}).dropna()

# === 3️⃣ 計算技術指標 ===
daily['MA50'] = daily['Close'].rolling(window=50).mean()

rolling_mean = daily['Close'].rolling(window=20).mean()
rolling_std = daily['Close'].rolling(window=20).std()
daily['Bollinger_upper'] = rolling_mean + 2 * rolling_std
daily['Bollinger_lower'] = rolling_mean - 2 * rolling_std

delta = daily['Close'].diff()
gain = delta.where(delta > 0, 0).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
daily['RSI'] = 100 - (100 / (1 + rs))

daily['Prev_Close'] = daily['Close'].shift(1)
daily['TR'] = daily[['High', 'Low', 'Prev_Close']].apply(
    lambda row: max(row['High'] - row['Low'], abs(row['High'] - row['Prev_Close']), abs(row['Low'] - row['Prev_Close'])),
    axis=1
)
daily['ATR'] = daily['TR'].rolling(window=14).mean()

daily['dist_MA50'] = abs(daily['Close'] - daily['MA50']) / daily['MA50']

cond_boll = daily['Close'] < daily['Bollinger_lower']
cond_rsi = daily['RSI'] < 30
cond_ma50 = daily['dist_MA50'] <= 0.03
daily['Long_signal'] = cond_boll & cond_rsi & cond_ma50

# === 4️⃣ 樣本內 / 外標記 ===
daily['sample'] = 'none'
daily.loc[(daily.index >= '2010-01-01') & (daily.index <= '2019-12-31'), 'sample'] = 'in'
daily.loc[(daily.index >= '2020-01-01') & (daily.index <= '2025-06-30'), 'sample'] = 'out'

# === 5️⃣ 回測邏輯 ===
capital = 1_000_000
capital_history = []
position = 0
entry_price = 0
fee_per_roundtrip = 1200
multiplier = 200

portfolio_value = []
returns = []
dates = []

for date, row in daily.iterrows():
    if position == 0 and row['Long_signal']:
        position = 1
        entry_price = row['Close']
        capital -= fee_per_roundtrip
    elif position == 1:
        stop_loss_price = entry_price - 2 * row['ATR']
        if row['Close'] <= stop_loss_price:
            pnl = (row['Close'] - entry_price) * multiplier
            capital += pnl
            capital -= fee_per_roundtrip
            position = 0
        elif date == daily.index[-1]:
            pnl = (row['Close'] - entry_price) * multiplier
            capital += pnl
            capital -= fee_per_roundtrip
            position = 0
    portfolio_value.append(capital)
    dates.append(date)
    returns.append(0 if len(portfolio_value) < 2 else (portfolio_value[-1] - portfolio_value[-2]) / portfolio_value[-2])

daily = daily.loc[dates]
daily['Portfolio_value'] = portfolio_value
daily['Return'] = returns

# === 6️⃣ 績效計算函數 ===
def perf_metrics(subset):
    cum_ret = (subset['Portfolio_value'].iloc[-1] / 1_000_000 - 1) * 100
    daily_ret = subset['Return']
    ann_ret = daily_ret.mean() * 252 * 100
    ann_vol = daily_ret.std() * np.sqrt(252) * 100
    mdd = ((subset['Portfolio_value'].cummax() - subset['Portfolio_value']) / subset['Portfolio_value'].cummax()).max() * 100
    sharpe = (ann_ret / ann_vol) if ann_vol > 0 else np.nan
    risk_ratio = (cum_ret / mdd) if mdd > 0 else np.nan
    return cum_ret, ann_ret, ann_vol, mdd, sharpe, risk_ratio

# === 7️⃣ 計算三段績效 ===
perf_all = perf_metrics(daily)
perf_in = perf_metrics(daily[daily['sample'] == 'in'])
perf_out = perf_metrics(daily[daily['sample'] == 'out'])

print("\n=== 全樣本 ===")
print(f"累積報酬: {perf_all[0]:.2f}%, 年化報酬: {perf_all[1]:.2f}%, 年化波動: {perf_all[2]:.2f}%, MDD: {perf_all[3]:.2f}%, Sharpe: {perf_all[4]:.2f}, 風報比: {perf_all[5]:.2f}")

print("\n=== 樣本內 ===")
print(f"累積報酬: {perf_in[0]:.2f}%, 年化報酬: {perf_in[1]:.2f}%, 年化波動: {perf_in[2]:.2f}%, MDD: {perf_in[3]:.2f}%, Sharpe: {perf_in[4]:.2f}, 風報比: {perf_in[5]:.2f}")

print("\n=== 樣本外 ===")
print(f"累積報酬: {perf_out[0]:.2f}%, 年化報酬: {perf_out[1]:.2f}%, 年化波動: {perf_out[2]:.2f}%, MDD: {perf_out[3]:.2f}%, Sharpe: {perf_out[4]:.2f}, 風報比: {perf_out[5]:.2f}")

# === 8️⃣ 每年報酬 / MDD ===
daily['Year'] = daily.index.year
annual_stats = daily.groupby('Year').apply(lambda x: pd.Series({
    'Annual Return (%)': (x['Portfolio_value'].iloc[-1] / x['Portfolio_value'].iloc[0] - 1) * 100,
    'Annual MDD (%)': ((x['Portfolio_value'].cummax() - x['Portfolio_value']) / x['Portfolio_value'].cummax()).max() * 100
}))

print("\n=== 每年報酬與 MDD ===")
print(annual_stats)

# === 9️⃣ 權益曲線圖 ===
plt.figure(figsize=(10,6))
plt.plot(daily.index, daily['Portfolio_value'], label='Equity Curve')
plt.axvline(pd.to_datetime('2019-12-31'), color='red', linestyle='--', label='Sample Split')
plt.xlabel('Date')
plt.ylabel('Portfolio Value (TWD)')
plt.title('Equity Curve with Sample Split')
plt.legend()
plt.grid()
plt.show()
