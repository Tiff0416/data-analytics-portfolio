import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import optuna
import os

# === 1. 讀取並聚合資料 ===
file_path = '/Users/tiffanykuo/Desktop/TMBA/TXF_R1_1min_data_combined.csv'
df = pd.read_csv(file_path, parse_dates=['datetime'])
df.set_index('datetime', inplace=True)

daily = df.resample('1D').agg({
    'Open': 'first',
    'High': 'max',
    'Low': 'min',
    'Close': 'last',
    'Volume': 'sum'
}).dropna()

daily['Prev_Close'] = daily['Close'].shift(1)
daily['sample'] = 'none'
daily.loc[(daily.index >= '2010-01-01') & (daily.index <= '2019-12-31'), 'sample'] = 'in'
daily.loc[(daily.index >= '2020-01-01') & (daily.index <= '2025-06-30'), 'sample'] = 'out'

# === 2. 回測函數（含嚴格空頭） ===
def run_backtest_with_strict_short(rsi_thresh=45, dist_thresh=0.15, atr_stop=3.0, score_thresh=3, return_df=False):
    d = daily.copy()
    d['MA20'] = d['Prev_Close'].rolling(window=20).mean()
    std = d['Prev_Close'].rolling(window=20).std()
    d['Bollinger_upper'] = d['MA20'] + 2 * std
    d['Bollinger_lower'] = d['MA20'] - 2 * std

    delta = d['Prev_Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    d['RSI'] = 100 - (100 / (1 + rs))

    d['TR'] = d[['High', 'Low', 'Prev_Close']].apply(
        lambda row: max(row['High'] - row['Low'], abs(row['High'] - row['Prev_Close']), abs(row['Low'] - row['Prev_Close'])),
        axis=1
    )
    d['ATR'] = d['TR'].rolling(window=14).mean().bfill()
    d['dist_MA20'] = abs(d['Prev_Close'] - d['MA20']) / d['MA20']
    d['MA50'] = d['Prev_Close'].rolling(window=50).mean()
    d['Trend_up'] = d['MA20'] > d['MA50']
    d['Trend_down'] = d['MA20'] < d['MA50']

    d['long_score'] = 0
    d.loc[d['Prev_Close'] < d['Bollinger_lower'], 'long_score'] += 1
    d.loc[d['RSI'] < rsi_thresh, 'long_score'] += 1
    d.loc[d['dist_MA20'] <= dist_thresh, 'long_score'] += 1
    d.loc[d['Trend_up'], 'long_score'] += 1

    d['short_score'] = 0
    d.loc[d['Prev_Close'] > d['Bollinger_upper'], 'short_score'] += 1
    d.loc[d['RSI'] > (100 - rsi_thresh), 'short_score'] += 1
    d.loc[d['dist_MA20'] <= dist_thresh, 'short_score'] += 1
    d.loc[d['Trend_down'], 'short_score'] += 1

    capital = 1_000_000
    position = 0
    entry_price = 0
    position_size = 0
    fee = 1200
    multiplier = 200
    portfolio_value, returns, dates = [], [], []
    hold_days = 0

    for i in range(1, len(d)):
        date = d.index[i]
        row = d.iloc[i]
        long_ps = 1.0 if d.iloc[i - 1]['long_score'] >= score_thresh else 0.0
        short_ps = 1.0 if (d.iloc[i - 1]['short_score'] == 4 and d.iloc[i - 1]['Trend_down']) else 0.0

        if position == 0:
            if long_ps > 0:
                position = 1
                entry_price = row['Open']
                position_size = long_ps
                hold_days = 0
                capital -= fee
            elif short_ps > 0:
                position = -1
                entry_price = row['Open']
                position_size = short_ps
                hold_days = 0
                capital -= fee

        elif position != 0:
            hold_days += 1
            if position == 1:
                stop_price = entry_price - atr_stop * row['ATR']
                if row['Close'] <= stop_price or hold_days >= 20:
                    pnl = (row['Close'] - entry_price) * multiplier * position_size
                    capital += pnl - fee
                    position = 0
            elif position == -1:
                stop_price = entry_price + atr_stop * row['ATR']
                if row['Close'] >= stop_price or hold_days >= 20:
                    pnl = (entry_price - row['Close']) * multiplier * position_size
                    capital += pnl - fee
                    position = 0

        portfolio_value.append(capital)
        dates.append(date)
        returns.append(0 if len(portfolio_value) < 2 else (portfolio_value[-1] - portfolio_value[-2]) / portfolio_value[-2])

    d = d.loc[dates]
    d['Portfolio_value'] = portfolio_value
    d['Return'] = returns

    def perf(subset):
        cum_ret = (subset['Portfolio_value'].iloc[-1] / 1_000_000 - 1) * 100
        daily_ret = subset['Return']
        ann_ret = daily_ret.mean() * 252 * 100
        ann_vol = daily_ret.std() * np.sqrt(252) * 100
        mdd = ((subset['Portfolio_value'].cummax() - subset['Portfolio_value']) / subset['Portfolio_value'].cummax()).max() * 100
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        risk_ratio = cum_ret / mdd if mdd > 0 else 0
        return cum_ret, ann_ret, ann_vol, mdd, sharpe, risk_ratio

    perf_in = perf(d[d['sample'] == 'in'])
    perf_out = perf(d[d['sample'] == 'out'])

    if return_df:
        return d, perf_in, perf_out
    else:
        return perf_in[4] - 0.3 * abs(perf_in[4] - perf_out[4])

# === 3. Optuna 最佳化（固定 random seed）===
def run_optuna():
    def objective(trial):
        rsi_thresh = trial.suggest_int("rsi_thresh", 30, 60)
        dist_thresh = trial.suggest_float("dist_thresh", 0.05, 0.3)
        atr_stop = trial.suggest_float("atr_stop", 1.5, 4.0)
        score_thresh = trial.suggest_int("score_thresh", 2, 4)
        return run_backtest_with_strict_short(rsi_thresh, dist_thresh, atr_stop, score_thresh)

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=30)
    return study.best_params

# === 4. 主程式 ===
if __name__ == '__main__':
    best_params = run_optuna()
    print("\n✅ 最佳參數:", best_params)

    df_final, perf_in, perf_out = run_backtest_with_strict_short(**best_params, return_df=True)

    print("\n=== 樣本內表現 ===")
    print(f"累積報酬: {perf_in[0]:.2f}%, 年化報酬: {perf_in[1]:.2f}%, 年化波動: {perf_in[2]:.2f}%, MDD: {perf_in[3]:.2f}%, Sharpe: {perf_in[4]:.2f}, 風報比: {perf_in[5]:.2f}")

    print("\n=== 樣本外表現 ===")
    print(f"累積報酬: {perf_out[0]:.2f}%, 年化報酬: {perf_out[1]:.2f}%, 年化波動: {perf_out[2]:.2f}%, MDD: {perf_out[3]:.2f}%, Sharpe: {perf_out[4]:.2f}, 風報比: {perf_out[5]:.2f}")

    # === 儲存圖表 ===
    plt.figure(figsize=(10, 6))
    plt.plot(df_final.index, df_final['Portfolio_value'], label='Equity Curve')
    plt.axvline(pd.to_datetime('2019-12-31'), color='red', linestyle='--', label='Sample Split')
    plt.title('Optimized Equity Curve with Strict Short Strategy')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.grid()
    plt.tight_layout()

    output_path = '/Users/tiffanykuo/Desktop/tmba/pic/optimized_equity_curve_strict_short.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"\n📈 圖表已儲存為 {output_path}")

    # === 新增進場得分欄位（entry_score） ===
    df_final['entry_score'] = 0
    df_final.loc[df_final['long_score'] >= best_params['score_thresh'], 'entry_score'] = df_final['long_score']
    df_final.loc[(df_final['short_score'] >= best_params['score_thresh']) & (df_final['Trend_down']), 'entry_score'] = -df_final['short_score']

    df_export = df_final[df_final['entry_score'] != 0][['Return', 'entry_score']].rename(columns={'Return': 'return'})

    csv_path = '/Users/tiffanykuo/Desktop/TMBA/visual pic/entry_return_data.csv'
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df_export.to_csv(csv_path, index=False)
    print(f'✅ 資料儲存於 {csv_path}')


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# 讀取資料
df = pd.read_csv('/Users/tiffanykuo/Desktop/TMBA/visual pic/entry_return_data.csv')

# 清理 entry_score 為數值型態
df['entry_score'] = pd.to_numeric(df['entry_score'], errors='coerce')

# 建立輸出資料夾
output_dir = '/Users/tiffanykuo/Desktop/TMBA/visual pic'
os.makedirs(output_dir, exist_ok=True)

# --- 1. Boxplot ---
plt.figure(figsize=(8, 5))
df.boxplot(column='return', by='entry_score', grid=True, showmeans=True)
plt.title('Return Distribution by Entry Score')
plt.suptitle('')
plt.xlabel('Entry Score')
plt.ylabel('Return')
plt.tight_layout()
plt.savefig(f'{output_dir}/boxplot_by_score.png')
plt.close()

# --- 2. Mean Return ---
mean_returns = df.groupby('entry_score')['return'].mean()
mean_returns.sort_index().plot(kind='bar', color='skyblue', grid=True)
plt.title('Mean Return by Entry Score')
plt.xlabel('Entry Score')
plt.ylabel('Mean Return')
plt.tight_layout()
plt.savefig(f'{output_dir}/mean_return_by_score.png')
plt.close()

# --- 3. Sharpe Ratio ---
sharpe_df = df.groupby('entry_score')['return'].agg(['mean', 'std'])
sharpe_df['sharpe'] = sharpe_df['mean'] / sharpe_df['std']
sharpe_df = sharpe_df.replace([np.inf, -np.inf], np.nan).dropna()
sharpe_df['sharpe'].sort_index().plot(kind='bar', color='lightgreen', grid=True)
plt.title('Sharpe Ratio by Entry Score')
plt.xlabel('Entry Score')
plt.ylabel('Sharpe Ratio')
plt.tight_layout()
plt.savefig(f'{output_dir}/sharpe_by_score.png')
plt.close()

# --- 4. Histogram per Score ---
unique_scores = df['entry_score'].dropna().unique()
for score in sorted(unique_scores):
    plt.figure()
    subset = df[df['entry_score'] == score]['return']
    plt.hist(subset, bins=20, color='orange', edgecolor='black')
    plt.axvline(subset.mean(), color='red', linestyle='dashed', linewidth=1, label=f'Mean: {subset.mean():.4f}')
    plt.title(f'Return Histogram (Score {int(score)})')
    plt.xlabel('Return')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/hist_score_{int(score)}.png')
    plt.close()


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# 讀取資料
df = pd.read_csv('/Users/tiffanykuo/Desktop/TMBA/visual pic/entry_return_data.csv')

# 若沒有日期，建立模擬日期
if 'date' not in df.columns:
    df['date'] = pd.date_range(start='2010-01-01', periods=len(df), freq='D')

df['cum_return'] = (1 + df['return']).cumprod()

plt.figure(figsize=(10, 6))
plt.plot(df['date'], df['cum_return'], label='Strategy Cumulative Return')
plt.title('Cumulative Return of the Strategy')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('/Users/tiffanykuo/Desktop/TMBA/visual pic/cumulative_return.png')
plt.close()

df['long_return'] = np.where(df['entry_score'] > 0, df['return'], 0)
df['short_return'] = np.where(df['entry_score'] < 0, df['return'], 0)

df['cum_long'] = (1 + df['long_return']).cumprod()
df['cum_short'] = (1 + df['short_return']).cumprod()

plt.figure(figsize=(10, 6))
plt.plot(df['date'], df['cum_long'], label='Long Only', color='green')
plt.plot(df['date'], df['cum_short'], label='Short Only', color='red')
plt.title('Long vs Short Cumulative Returns')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('/Users/tiffanykuo/Desktop/TMBA/visual pic/long_vs_short.png')
plt.close()

plt.figure(figsize=(10, 6))
colors = np.where(df['entry_score'] > 0, 'green', 'red')
plt.scatter(range(len(df)), df['return'], c=colors, alpha=0.6)
plt.axhline(0, color='gray', linestyle='--')
plt.title('PnL per Trade')
plt.xlabel('Trade Index')
plt.ylabel('Return')
plt.tight_layout()
plt.savefig('/Users/tiffanykuo/Desktop/TMBA/visual pic/pnl_scatter.png')
plt.close()
