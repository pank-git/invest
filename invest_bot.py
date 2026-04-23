import os
import io
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import linregress
from datetime import datetime, timedelta

# Matplotlib setup for server/headless environments
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Telegram imports
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes

# ==========================================
# Configuration
# ==========================================
CSV_FILE_PATH = "symbols.csv"

# ==========================================
# Core Financial & Data Logic
# ==========================================

def get_stock_data(symbol, days=1825):
    end = datetime.now()
    start = end - timedelta(days=days)
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start, end=end, interval='1d')
        if df.empty:
            return None, None

        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
            
        df = df[['Close']].copy()
        name = ticker.info.get('shortName', symbol)
        return df, name
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None, None

def calculate_sma(df):
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_150'] = df['Close'].rolling(window=150).mean()
    return df

def get_sma_trend(df, sma_col='SMA_50', lookback=10):
    recent = df[sma_col].dropna().tail(lookback)
    if len(recent) < 2:
        return "Insufficient data"
    
    x = np.arange(len(recent))
    slope, _, _, _, _ = linregress(x, recent)
    price_level = recent.iloc[-1]
    threshold = 0.001 * price_level
    
    if slope > threshold: return "Upward"
    elif slope < -threshold: return "Downward"
    else: return "Flat"

def get_symbols_from_csv(filepath):
    """Robust helper to extract symbols from every row of the CSV."""
    if not os.path.exists(filepath):
        return []
    symbols = []
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        lines = f.read().splitlines()
    for line in lines:
        if not line.strip(): continue
        parts = [p.strip() for p in line.split(',')]
        if parts[0].lower() in ['symbol', 'ticker', 'symbols']: continue
        if len(lines) == 1:
            for part in parts:
                if part: symbols.append(part)
        else:
            if parts[0]: symbols.append(parts[0])
    return list(dict.fromkeys(symbols))

def get_summary_data(content):
    """Parses CSV content and returns structured data for reports."""
    symbols = get_symbols_from_csv(CSV_FILE_PATH)
    results = []
    for symbol in symbols:
        df, name = get_stock_data(symbol, days=250)
        if df is None: continue
        df = calculate_sma(df)
        latest_df = df.dropna()
        if len(latest_df) < 4: continue
        
        latest = latest_df.iloc[-1]
        prev_day = latest_df.iloc[-2]
        
        status = "BULL" if latest['Close'] > latest['SMA_50'] > latest['SMA_150'] else \
                 "BEAR" if latest['Close'] < latest['SMA_50'] < latest['SMA_150'] else "NEUT"
        cross = "GOLD" if latest['SMA_50'] > latest['SMA_150'] and prev_day['SMA_50'] <= prev_day['SMA_150'] else \
                "DEATH" if latest['SMA_50'] < latest['SMA_150'] and prev_day['SMA_50'] >= prev_day['SMA_150'] else "-"
        
        hist = df['Close'].tail(3).values
        hist_str = "/".join([f"{x:.2f}" for x in hist])
        price_change = "UP" if latest['Close'] >= prev_day['Close'] else "DOWN"
        
        results.append({
            'Ticker': symbol[:6], 'Name': name, 'Status': status,
            'Cross': cross, 'Price': f"{latest['Close']:.2f}",
            'Last 3 Days': hist_str, 'PriceChange': price_change
        })
    return results

# ==========================================
# Formatting Engines
# ==========================================

def format_text_table(data):
    if not data: return "No data found."
    output = "<b>📊 Stock Summary (Text)</b>\n<code>"
    output += "Tickr|Name        |Stat |Cross|Price |Last 3 Days\n"
    output += "-" * 55 + "\n"
    for row in data:
        short_name = (row['Name'][:10] + "..") if len(row['Name']) > 12 else f"{row['Name']:<12}"
        output += f"{row['Ticker']:<5}|{short_name[:12]:<12}|{row['Status']:<5}|{row['Cross']:<5}|{row['Price']:>6}|{row['Last 3 Days']}\n"
    output += "</code>"
    return output

def format_image_table(data):
    if not data: return None
    raw_df = pd.DataFrame(data)
    display_df = raw_df.drop(columns=['PriceChange'])
    fig_height = max(2, len(data) * 0.4 + 1)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.axis('tight'); ax.axis('off')
    
    table = ax.table(cellText=display_df.values, colLabels=display_df.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False); table.set_fontsize(11); table.scale(1.2, 1.8)
    
    for (row_idx, col_idx), cell in table.get_celld().items():
        if row_idx == 0:
            cell.set_text_props(weight='bold', color='white'); cell.set_facecolor('#2c3e50')
        elif row_idx > 0:
            row_data = data[row_idx - 1]
            if col_idx == 2: # Status
                if row_data['Status'] == "BULL": cell.set_text_props(color='green', weight='bold')
                if row_data['Status'] == "BEAR": cell.set_text_props(color='red', weight='bold')
            if col_idx == 4: # Price
                p_color = 'green' if row_data['PriceChange'] == "UP" else 'red'
                cell.set_text_props(color=p_color, weight='bold')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0); plt.close()
    return buf

def plot_price_sma(df, symbol):
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Close'], color='black', label='Price', linewidth=1.5)
    plt.plot(df.index, df['SMA_50'], color='blue', label='SMA 50', linewidth=2)
    plt.plot(df.index, df['SMA_150'], color='green', label='SMA 150', linewidth=2)
    
    df = df.copy()
    df['SMA_Cross'] = 0
    df.loc[(df['SMA_50'] > df['SMA_150']) & (df['SMA_50'].shift(1) <= df['SMA_150'].shift(1)), 'SMA_Cross'] = 1 
    df.loc[(df['SMA_50'] < df['SMA_150']) & (df['SMA_50'].shift(1) >= df['SMA_150'].shift(1)), 'SMA_Cross'] = -1 
    
    cross_points = df[df['SMA_Cross'] != 0]
    for date, row in cross_points.iterrows():
        direction = 'GOLD' if row['SMA_Cross'] == 1 else 'DEATH'
        plt.scatter(date, row['SMA_50'], color='red', marker='o', s=80, zorder=5)
        plt.annotate(direction, xy=(date, row['SMA_50']), xytext=(5, 5), textcoords='offset points', fontsize=8, fontweight='bold', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    plt.title(f'{symbol} Price vs SMAs'); plt.legend(); plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45); plt.tight_layout()
    buf = io.BytesIO(); plt.savefig(buf, format='png'); buf.seek(0); plt.close()
    return buf

# ==========================================
# Telegram Bot Handlers
# ==========================================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("📈 *Stock SMA Bot*\n/analyze [Ticker]\n/analyze_list\n/summary\n/summary_img\n/cancel", parse_mode='Markdown')

async def track_price_vs_sma(symbol):
    df, name = get_stock_data(symbol)
    if df is None: return f"No data for {symbol}", None
    df = calculate_sma(df)
    latest = df.dropna().iloc[-1]
    report = f"<b>{name} ({symbol})</b>\nPrice: ${latest['Close']:.2f}\nSMA50: {get_sma_trend(df, 'SMA_50')}\nSMA150: {get_sma_trend(df, 'SMA_150')}"
    return report, df

async def analyze_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args: return
    symbol = context.args[0].upper()
    report, df = await track_price_vs_sma(symbol)
    if df is not None:
        await update.message.reply_text(report, parse_mode='HTML')
        await update.message.reply_photo(plot_price_sma(df, symbol))

async def summary_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    data = get_summary_data(None)
    await update.message.reply_text(format_text_table(data), parse_mode='HTML')

async def summary_img_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("⏳ Generating image...")
    data = get_summary_data(None)
    img = format_image_table(data)
    if img: await update.message.reply_photo(img, caption="📈 Market Analysis Summary")

async def analyze_list_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    symbols = get_symbols_from_csv(CSV_FILE_PATH)
    if not symbols: return
    context.user_data['symbol_list'] = symbols
    msg = "📋 *Select Symbol:*\n" + "\n".join([f"{i+1}. {s}" for i, s in enumerate(symbols)])
    await update.message.reply_text(msg, parse_mode='Markdown')

async def handle_user_reply(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if 'symbol_list' not in context.user_data: return
    text = update.message.text.strip()
    symbols = context.user_data['symbol_list']
    if text.isdigit() and 1 <= int(text) <= len(symbols):
        symbol = symbols[int(text) - 1]
        del context.user_data['symbol_list']
        report, df = await track_price_vs_sma(symbol)
        await update.message.reply_text(report, parse_mode='HTML')
        await update.message.reply_photo(plot_price_sma(df, symbol))

async def cancel_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.pop('symbol_list', None)
    await update.message.reply_text("Canceled.")

if __name__ == "__main__":
    BOT_TOKEN = os.getenv("PANKINVEST_BOT_TOKEN")
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("analyze", analyze_command))
    app.add_handler(CommandHandler("analyze_list", analyze_list_command))
    app.add_handler(CommandHandler("summary", summary_command))
    app.add_handler(CommandHandler("summary_img", summary_img_command))
    app.add_handler(CommandHandler("cancel", cancel_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_user_reply))
    print("Bot is running. Press Ctrl+C to stop.")
    app.run_polling()
