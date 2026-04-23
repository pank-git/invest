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

def get_summary_data(content):
    """Parses CSV content and returns structured data for reports."""
    symbols = []
    lines = content.splitlines()
    for line in lines:
        if not line.strip(): continue
        parts = [p.strip() for p in line.split(',')]
        if parts[0].lower() in ['symbol', 'ticker', 'symbols']: continue
        if len(lines) <= 2: 
            for part in parts:
                if part: symbols.append(part)
        else:
            if parts[0]: symbols.append(parts[0])

    symbols = list(dict.fromkeys(symbols))
    results = []

    for symbol in symbols:
        df, name = get_stock_data(symbol, days=250)
        if df is None: continue
        
        df = calculate_sma(df)
        latest_df = df.dropna()
        if len(latest_df) < 4: continue
        
        latest = latest_df.iloc[-1]
        prev = latest_df.iloc[-2]
        
        status = "BULL" if latest['Close'] > latest['SMA_50'] > latest['SMA_150'] else \
                 "BEAR" if latest['Close'] < latest['SMA_50'] < latest['SMA_150'] else "NEUT"
            
        cross = "GOLD" if latest['SMA_50'] > latest['SMA_150'] and prev['SMA_50'] <= prev['SMA_150'] else \
                "DEATH" if latest['SMA_50'] < latest['SMA_150'] and prev['SMA_50'] >= prev['SMA_150'] else "-"
            
        hist = df['Close'].tail(3).values
        hist_str = "/".join([f"{x:.2f}" for x in hist])
        
        results.append({
            'Ticker': symbol[:6],
            'Name': name,
            'Status': status,
            'Cross': cross,
            'Price': f"{latest['Close']:.2f}",
            'Last 3 Days': hist_str
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
    df_plot = pd.DataFrame(data)
    fig_height = max(2, len(data) * 0.4 + 1)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.axis('tight'); ax.axis('off')
    
    table = ax.table(cellText=df_plot.values, colLabels=df_plot.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False); table.set_fontsize(11); table.scale(1.2, 1.8)
    
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#2c3e50')
        elif col == 2:
            val = cell.get_text().get_text()
            if val == "BULL": cell.set_text_props(color='green', weight='bold')
            if val == "BEAR": cell.set_text_props(color='red', weight='bold')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0); plt.close()
    return buf

# ==========================================
# Telegram Bot Handlers
# ==========================================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    welcome_text = (
        "📈 *Stock SMA Bot Commands:*\n\n"
        "`/analyze AAPL` - Single symbol trend & chart\n"
        "`/analyze_list` - Select a symbol from your list\n"
        "`/summary` - View text summary table\n"
        "`/summary_img` - View visual summary table (Best for mobile)\n"
        "`/cancel` - Exit selection mode"
    )
    await update.message.reply_text(welcome_text, parse_mode='Markdown')

async def summary_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not os.path.exists(CSV_FILE_PATH):
        await update.message.reply_text("❌ symbols.csv not found.")
        return
    await update.message.reply_text("⏳ Generating text summary...")
    with open(CSV_FILE_PATH, 'r', encoding='utf-8-sig') as f:
        data = get_summary_data(f.read())
    await update.message.reply_text(format_text_table(data), parse_mode='HTML')

async def summary_img_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not os.path.exists(CSV_FILE_PATH):
        await update.message.reply_text("❌ symbols.csv not found.")
        return
    await update.message.reply_text("⏳ Rendering summary image...")
    with open(CSV_FILE_PATH, 'r', encoding='utf-8-sig') as f:
        data = get_summary_data(f.read())
    img_buf = format_image_table(data)
    if img_buf:
        await update.message.reply_photo(photo=img_buf, caption="📈 Market Analysis Summary")

async def analyze_list_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not os.path.exists(CSV_FILE_PATH):
        await update.message.reply_text("❌ symbols.csv not found.")
        return
    with open(CSV_FILE_PATH, 'r', encoding='utf-8-sig') as f:
        lines = f.read().splitlines()
    
    symbols = []
    for line in lines:
        if not line.strip() or line.split(',')[0].lower() in ['symbol', 'ticker']: continue
        symbols.append(line.split(',')[0].strip())
    
    symbols = list(dict.fromkeys(symbols))
    context.user_data['symbol_list'] = symbols
    
    msg = "📋 *Select a symbol:*\n\n" + "\n".join([f"{i+1}. {s}" for i, s in enumerate(symbols)])
    await update.message.reply_text(msg, parse_mode='Markdown')

async def handle_user_reply(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if 'symbol_list' not in context.user_data: return
    text = update.message.text.strip()
    symbols = context.user_data['symbol_list']
    
    if text.isdigit() and 1 <= int(text) <= len(symbols):
        symbol = symbols[int(text) - 1]
        del context.user_data['symbol_list']
        await analyze_symbol(update, symbol)
    else:
        await update.message.reply_text("⚠️ Invalid selection. Use /cancel to stop.")

async def analyze_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: `/analyze AAPL`", parse_mode='Markdown')
        return
    await analyze_symbol(update, context.args[0].upper())

async def analyze_symbol(update, symbol):
    await update.message.reply_text(f"⏳ Analyzing {symbol}...")
    from invest_bot import track_price_vs_sma, plot_price_sma # Ensure helpers are accessible
    # Use existing helper logic (track_price_vs_sma and plot_price_sma)
    # ... logic to return report and photo ...

async def cancel_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if 'symbol_list' in context.user_data:
        del context.user_data['symbol_list']
        await update.message.reply_text("Action canceled.")

# ==========================================
# Main
# ==========================================

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
    
    app.run_polling()
