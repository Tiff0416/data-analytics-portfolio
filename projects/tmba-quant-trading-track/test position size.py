import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === 讀取資料與設定 ===
file_path = '/Users/tiffanykuo/Desktop/TMBA/TXF_R1_1min_data_combined.csv'
df = pd.read_csv(file_path, parse_dates=['datetime'])
df.set_index('datetime', inplace=True)

# === 日資料聚合 ===
daily = df.resample('1D').agg({
    'Open': 'first',
    'High': 'max',
    'Low': 'min',
    'Close': 'last',
    'Volume': 'sum'
}).dropna()

# === 樣本區分 ===
daily['sample'] = 'none'
daily.loc[(daily.index >= '2010-01-01') & (daily.index <= '2019-12-31'), 'sample'] = 'in'
daily.loc[(daily.index >= '2020-01-01') & (daily.index <= '2025-06-30'), 'sample'] = 'out'

# === 技術指標計算 ===
daily['MA20'] = daily['Close'].rolling(20).mean()
daily['MA50'] = daily['Close'].rolling(50).mean()
daily['STD20'] = daily['Close'].rolling(20).std()
daily['Upper'] = daily['MA20'] + 2 * daily['STD20']
daily['Lower'] = daily['MA20'] - 2 * daily['STD20']

delta = daily['Close'].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()
rs = avg_gain / avg_loss
daily['RSI'] = 100 - (100 / (1 + rs))
daily['ATR'] = (daily['High'] - daily['Low']).rolling(14).mean()

# === 訊號評分 ===
daily['Trend_down'] = daily['MA20'] < daily['MA50']
daily['long_score'] = 0
daily['short_score'] = 0

daily.loc[daily['Close'] < daily['Lower'], 'long_score'] += 1
daily.loc[daily['RSI'] < 40, 'long_score'] += 1
daily.loc[(daily['MA20'] - daily['Close']) / daily['MA20'] > 0.03, 'long_score'] += 1
daily.loc[~daily['Trend_down'], 'long_score'] += 1

daily.loc[daily['Close'] > daily['Upper'], 'short_score'] += 1
daily.loc[daily['RSI'] > 60, 'short_score'] += 1
daily.loc[(daily['Close'] - daily['MA20']) / daily['MA20'] > 0.03, 'short_score'] += 1
daily.loc[daily['Trend_down'], 'short_score'] += 1

# === 回測設定 ===
daily['return'] = daily['Close'].pct_change()
daily['position_size'] = 0.0
daily['pnl'] = 0.0
daily['cumulative_return'] = 1.0
daily['position_type'] = 'none'

score_thresh = 3
initial_capital = 1_000_000
fee_per_trade = 150 / initial_capital

# === 回測主邏輯 ===
for i in range(1, len(daily)):
    row = daily.iloc[i - 1]
    ret = daily.iloc[i]['return']

    long_score = row['long_score']
    short_score = row['short_score']
    trend_down = row['Trend_down']

    long_ps = (long_score / 4.0) if long_score >= score_thresh else 0.0
    short_ps = (short_score / 4.0) if short_score >= score_thresh and trend_down else 0.0

    if long_ps > short_ps:
        position_size = long_ps
        position_type = 'long'
        pnl = ret * position_size - fee_per_trade
    elif short_ps > long_ps:
        position_size = short_ps
        position_type = 'short'
        pnl = -ret * position_size - fee_per_trade
    else:
        position_size = 0.0
        position_type = 'none'
        pnl = 0.0

    daily.iloc[i, daily.columns.get_loc('position_size')] = position_size
    daily.iloc[i, daily.columns.get_loc('pnl')] = pnl
    daily.iloc[i, daily.columns.get_loc('cumulative_return')] = daily.iloc[i - 1]['cumulative_return'] * (1 + pnl)
    daily.iloc[i, daily.columns.get_loc('position_type')] = position_type

# 儲存回測結果
daily.to_csv('/Users/tiffanykuo/Desktop/TMBA/backtest_result.csv')
print("回測完成，結果已儲存為 'backtest_result.csv'")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# === 1. 載入資料 ===
file_path = '/Users/tiffanykuo/Desktop/TMBA/backtest_result.csv'
df = pd.read_csv(file_path, parse_dates=['datetime'])
df.set_index('datetime', inplace=True)

# === 2. 技術指標計算與標準化打分 ===
df['MA20'] = df['Close'].rolling(window=20).mean()
df['MA50'] = df['Close'].rolling(window=50).mean()
df['Boll_upper'] = df['MA20'] + 2 * df['Close'].rolling(window=20).std()
df['Boll_lower'] = df['MA20'] - 2 * df['Close'].rolling(window=20).std()
df['RSI'] = 100 - (100 / (1 + df['Close'].diff().clip(lower=0).rolling(14).mean() / 
                             (-df['Close'].diff().clip(upper=0)).rolling(14).mean()))
df['dist_MA20'] = abs(df['Close'] - df['MA20']) / df['MA20']
df['Trend_up'] = df['MA20'] > df['MA50']
df['Trend_down'] = df['MA20'] < df['MA50']
df['boll_dist'] = (df['Close'] - df['MA20']) / (df['Boll_upper'] - df['Boll_lower'])

# Z-score 標準化
df['z_rsi'] = (df['RSI'] - df['RSI'].rolling(window=20).mean()) / df['RSI'].rolling(window=20).std()
df['z_dist'] = (df['dist_MA20'] - df['dist_MA20'].rolling(window=20).mean()) / df['dist_MA20'].rolling(window=20).std()
df['z_boll'] = (df['boll_dist'] - df['boll_dist'].rolling(window=20).mean()) / df['boll_dist'].rolling(window=20).std()

# 統一 score 標準：做多 越小越好，做空 越大越好
df['long_score'] = (-df['z_rsi']) + (-df['z_dist']) + (-df['z_boll']) + df['Trend_up'].astype(int)
df['short_score'] = (df['z_rsi']) + (df['z_dist']) + (df['z_boll']) + df['Trend_down'].astype(int)

# === 3. 樣本標記 ===
df['sample'] = 'none'
df.loc[(df.index >= '2010-01-01') & (df.index <= '2019-12-31'), 'sample'] = 'in'
df.loc[(df.index >= '2020-01-01') & (df.index <= '2025-06-30'), 'sample'] = 'out'

# === 4. 權益曲線繪圖 ===
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['cumulative_return'], label='Equity Curve')
plt.axvline(pd.to_datetime('2019-12-31'), color='red', linestyle='--', label='Sample Split')
plt.title('Equity Curve (Sample In vs Out)')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("/Users/tiffanykuo/Desktop/TMBA/pic/equity_curve_split.png", dpi=300)
print("✅ 權益曲線已儲存")

# === 5. 績效指標計算 ===
def calculate_metrics(sub_df):
    daily_ret = sub_df['cumulative_return'].pct_change().dropna()
    cum_return = (sub_df['cumulative_return'].iloc[-1] / sub_df['cumulative_return'].iloc[0] - 1) * 100
    ann_return = daily_ret.mean() * 252 * 100
    ann_vol = daily_ret.std() * np.sqrt(252) * 100
    mdd = (sub_df['cumulative_return'].cummax() - sub_df['cumulative_return']).max() / sub_df['cumulative_return'].cummax().max() * 100
    sharpe = ann_return / ann_vol if ann_vol != 0 else 0
    risk_ratio = cum_return / mdd if mdd != 0 else 0
    return cum_return, ann_return, ann_vol, mdd, sharpe, risk_ratio

in_sample = df[df['sample'] == 'in']
out_sample = df[df['sample'] == 'out']
all_sample = df[df['sample'].isin(['in', 'out'])]

metrics = {
    '全樣本 (All)': calculate_metrics(all_sample),
    '樣本內 (In-Sample)': calculate_metrics(in_sample),
    '樣本外 (Out-of-Sample)': calculate_metrics(out_sample)
}

perf_df = pd.DataFrame(metrics, index=[
    '累積報酬 (%)', '年化報酬 (%)', '年化波動 (%)', '最大回落 MDD (%)', '年化夏普', '風報比 (%)'
]).T

print("\n📊 績效指標：")
print(perf_df)
