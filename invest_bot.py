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
# Set the name of the CSV file stored on your system
CSV_FILE_PATH = "symbols.csv"

# ==========================================
# Core Financial Logic
# ==========================================

def get_stock_data(symbol, days=1825):
    end = datetime.now()
    start = end - timedelta(days=days)
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start, end=end, interval='1d')
        if df.empty:
            return None, None
        df = df[['Close']].copy()
        name = ticker.info.get('longName', symbol)
        return df, name
    except Exception as e:
        print(e)
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

def track_price_vs_sma(symbol):
    df, name = get_stock_data(symbol)
    if df is None:
        return f"No data available for {symbol}", None
    
    df = calculate_sma(df)
    latest_df = df.dropna()
    if latest_df.empty:
        return f"Insufficient data for SMAs on {symbol}", None
    
    latest = latest_df.iloc[-1]
    
    report = f"<b>{name} ({symbol}) Analysis:</b>\n"
    report += f"Current Price: ${latest['Close']:.2f}\n"
    report += f"SMA 50: ${latest['SMA_50']:.2f} ({get_sma_trend(df, 'SMA_50')})\n"
    report += f"SMA 150: ${latest['SMA_150']:.2f} ({get_sma_trend(df, 'SMA_150')})\n\n"
    
    if latest['Close'] > latest['SMA_50'] > latest['SMA_150']:
        report += "Status: <b>Bullish</b> (Price > SMA50 > SMA150)\n"
    elif latest['Close'] < latest['SMA_50'] < latest['SMA_150']:
        report += "Status: <b>Bearish</b> (Price < SMA50 < SMA150)\n"
    else:
        report += "Status: <b>Neutral/Mixed</b>\n"
    
    return report, df

def plot_price_sma(df, symbol):
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Close'], color='black', label='Price', linewidth=1.5)
    plt.plot(df.index, df['SMA_50'], color='blue', label='SMA 50', linewidth=2)
    plt.plot(df.index, df['SMA_150'], color='green', label='SMA 150', linewidth=2)
    
    df['SMA_Cross'] = 0
    df.loc[(df['SMA_50'] > df['SMA_150']) & (df['SMA_50'].shift(1) <= df['SMA_150'].shift(1)), 'SMA_Cross'] = 1 
    df.loc[(df['SMA_50'] < df['SMA_150']) & (df['SMA_50'].shift(1) >= df['SMA_150'].shift(1)), 'SMA_Cross'] = -1 
    
    cross_points = df[df['SMA_Cross'] != 0].copy()
    if not cross_points.empty:
        for date, row in cross_points.iterrows():
            direction = 'Golden (Bullish)' if row['SMA_Cross'] == 1 else 'Death (Bearish)'
            date_str = date.strftime('%d-%b-%Y')
            plt.scatter(date, row['SMA_50'], color='red', marker='o', s=80, zorder=5, edgecolors='darkred', linewidth=1.5)
            offset_y = 15 if row['SMA_Cross'] == 1 else -15 
            plt.annotate(f"{direction[:4]}\n{date_str}", 
                        xy=(date, row['SMA_50']), xytext=(8, offset_y),
                        textcoords='offset points', fontsize=9, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8),
                        arrowprops=dict(arrowstyle='->', color='red', lw=1),
                        ha='left', va='center')
        
        plt.scatter([], [], color='red', marker='o', s=80, label='SMA Crossovers', zorder=5)
    else:
        plt.scatter([], [], color='red', marker='o', s=80, label='No recent crossovers')
    
    plt.title(f'{symbol} Price vs SMAs')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=4))
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save plot to a memory buffer to send via Telegram
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf

def process_csv_content(content):
    symbols = []
    lines = content.splitlines()
    for line in lines:
        if not line.strip(): continue
        parts = [p.strip() for p in line.split(',')]
        if parts[0].lower() in ['symbol', 'ticker', 'symbols']: continue
        
        if len(lines) == 1 or (len(lines) == 2 and lines[0].lower().startswith('symbol')):
            for part in parts:
                if part: symbols.append(part)
            break
        else:
            if parts[0]: symbols.append(parts[0])

    symbols = list(dict.fromkeys(symbols))
    if not symbols: return "No valid symbols found."

    # Added Name column and adjusted spacing to fit mobile screens
    output = "<pre>\n"
    output += "Sym    | Name         | Status  | Cross  | Price  | Last 3 Days\n"
    output += "-" * 70 + "\n"

    for symbol in symbols:
        # get_stock_data already fetches the name in the background
        df, name = get_stock_data(symbol, days=250)
        
        # Keep up to 6 characters for the symbol (e.g., 'U11.SI')
        sym_str = symbol[:6]
        
        if df is None:
            output += f"{sym_str:<6} | {'Error':<12} | Error   | -      | -      | -\n"
            continue
        
        # Format the Name: truncate to 12 characters so it doesn't break the table
        display_name = name if name != symbol else "Unknown"
        short_name = (display_name[:10] + '..') if len(display_name) > 12 else display_name[:12]
        
        df = calculate_sma(df)
        latest_df = df.dropna()
        if len(latest_df) < 2:
            output += f"{sym_str:<6} | {short_name:<12} | No Data | -      | -      | -\n"
            continue
        
        latest = latest_df.iloc[-1]
        previous = latest_df.iloc[-2]
        
        if latest['Close'] > latest['SMA_50'] > latest['SMA_150']: status = "Bullish"
        elif latest['Close'] < latest['SMA_50'] < latest['SMA_150']: status = "Bearish"
        else: status = "Neutral"
            
        if latest['SMA_50'] > latest['SMA_150'] and previous['SMA_50'] <= previous['SMA_150']:
            crossover = "Golden"
        elif latest['SMA_50'] < latest['SMA_150'] and previous['SMA_50'] >= previous['SMA_150']:
            crossover = "Death"
        else:
            crossover = "None"
            
        price = f"${latest['Close']:.2f}"
        last_3 = df['Close'].tail(3).apply(lambda x: f"{x:.1f}").tolist()
        last_3_str = ",".join(last_3)
        
        output += f"{sym_str:<6} | {short_name:<12} | {status[:7]:<7} | {crossover[:6]:<6} | {price:<6} | {last_3_str}\n"
    
    output += "</pre>"
    return output

# ==========================================
# Telegram Bot Handlers
# ==========================================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    welcome_text = (
        "📈 *Welcome to the Stock SMA Bot!*\n\n"
        "Commands:\n"
        "`/analyze AAPL` - Get current trend and SMA chart for a single symbol.\n"
        "`/summary` - Generate a summary table from the system's saved CSV file."
    )
    await update.message.reply_text(welcome_text, parse_mode='Markdown')

async def analyze_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Please provide a symbol. Example: `/analyze AAPL`")
        return
    
    symbol = context.args[0].upper()
    await update.message.reply_text(f"⏳ Analyzing {symbol}...")
    
    report, df = track_price_vs_sma(symbol)
    
    if df is not None:
        await update.message.reply_text(report, parse_mode='HTML')
        photo_buf = plot_price_sma(df, symbol)
        await update.message.reply_photo(photo=photo_buf)
    else:
        await update.message.reply_text(report)

async def summary_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Check if the CSV file exists on the local system
    if not os.path.exists(CSV_FILE_PATH):
        await update.message.reply_text(f"❌ Error: The file `{CSV_FILE_PATH}` was not found on the server.", parse_mode='Markdown')
        return
        
    await update.message.reply_text("⏳ Reading system file and generating summary... This might take a moment.")
    
    try:
        # Read the file locally
        with open(CSV_FILE_PATH, 'r', encoding='utf-8-sig') as f:
            content = f.read()
            
        # Process and send the report
        summary_report = process_csv_content(content)
        await update.message.reply_text(summary_report, parse_mode='HTML')
    except Exception as e:
        await update.message.reply_text(f"❌ Error processing file: {e}")
        
def get_symbols_from_csv(filepath):
    """Helper function to extract a clean list of symbols from the CSV."""
    if not os.path.exists(filepath):
        return []
    
    symbols = []
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        content = f.read()
        
    lines = content.splitlines()
    for line in lines:
        if not line.strip(): continue
        parts = [p.strip() for p in line.split(',')]
        if parts[0].lower() in ['symbol', 'ticker', 'symbols']: continue
        
        if len(lines) == 1 or (len(lines) == 2 and lines[0].lower().startswith('symbol')):
            for part in parts:
                if part: symbols.append(part)
            break
        else:
            if parts[0]: symbols.append(parts[0])

    return list(dict.fromkeys(symbols))

async def analyze_list_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Generates a numbered list of symbols and waits for a reply."""
    symbols = get_symbols_from_csv(CSV_FILE_PATH)
    
    if not symbols:
        await update.message.reply_text("❌ No symbols found in the system's CSV file.")
        return
        
    # Store the list in user_data so the bot remembers it for their next message
    context.user_data['symbol_list'] = symbols
    
    msg = "📋 *Select a symbol to analyze:*\n\n"
    for i, sym in enumerate(symbols, 1):
        msg += f"{i}. {sym}\n"
    
    msg += "\n_Reply with the number corresponding to the symbol._"
    await update.message.reply_text(msg, parse_mode='Markdown')

async def handle_user_reply(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Catches the user's numeric reply and runs the analysis."""
    # If the user isn't currently selecting a symbol, ignore their text
    if 'symbol_list' not in context.user_data:
        return
        
    text = update.message.text.strip()
    symbols = context.user_data['symbol_list']
    
    if text.isdigit():
        choice = int(text)
        
        if 1 <= choice <= len(symbols):
            symbol = symbols[choice - 1]
            
            # Clear the stored list so they aren't stuck in "selection mode"
            del context.user_data['symbol_list']
            
            await update.message.reply_text(f"⏳ Analyzing {symbol}...")
            
            # Run the exact same analysis logic
            report, df = track_price_vs_sma(symbol)
            if df is not None:
                await update.message.reply_text(report, parse_mode='HTML')
                photo_buf = plot_price_sma(df, symbol)
                await update.message.reply_photo(photo=photo_buf)
            else:
                await update.message.reply_text(report)
        else:
            await update.message.reply_text(f"⚠️ Please enter a number between 1 and {len(symbols)}.")
    else:
        await update.message.reply_text("⚠️ Please reply with a valid number from the list, or type /cancel to abort.")

async def cancel_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Allows the user to exit selection mode."""
    if 'symbol_list' in context.user_data:
        del context.user_data['symbol_list']
        await update.message.reply_text("Action canceled.")
    else:
        await update.message.reply_text("Nothing to cancel.")        
        

if __name__ == "__main__":
    # Read the token from the system environment variable
    BOT_TOKEN = os.getenv("PANKINVEST_BOT_TOKEN")
    
    # Safety check to ensure the token was found
    if not BOT_TOKEN:
        print("❌ Error: TELEGRAM_BOT_TOKEN environment variable is not set.")
        print("Please set it before running the bot.")
        exit(1)
    
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("analyze", analyze_command))
    app.add_handler(CommandHandler("summary", summary_command))
    
    # NEW HANDLERS
    app.add_handler(CommandHandler("analyze_list", analyze_list_command))
    app.add_handler(CommandHandler("cancel", cancel_command))
    
    # This catches regular text messages (the numbers) but ignores commands starting with "/"
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_user_reply))
    
    print("Bot is running. Press Ctrl+C to stop.")
    app.run_polling()
