import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta
from datetime import datetime, timedelta
import time

# Page config
st.set_page_config(page_title="AI Trading Bot", layout="wide", page_icon="🚀")

# Custom CSS
st.markdown("""
<style>
.main-header {font-size: 3rem; color: #1e40af; font-weight: bold;}
.metric-card {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 10px;}
.signal-card {padding: 1.5rem; border-radius: 15px; margin: 1rem 0;}
.buy-signal {background: linear-gradient(135deg, #10b981, #059669);}
.sell-signal {background: linear-gradient(135deg, #ef4444, #dc2626);}
.neutral-signal {background: linear-gradient(135deg, #6b7280, #4b5563);}
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=300, show_spinner=False)
def get_data_cached(symbol, period="30d", interval="1h"):
    """Fetch and prepare market data with error handling"""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period, interval=interval)
        if data.empty or len(data) < 50:
            return None
        return add_indicators(data)
    except Exception as e:
        st.error(f"Error fetching {symbol}: {str(e)}")
        return None

def add_indicators(data):
    """Add technical indicators to dataframe"""
    data = data.copy()
    data['RSI'] = ta.momentum.rsi(data['Close'], length=14)
    data['EMA20'] = ta.trend.ema_indicator(data['Close'], window=20)
    data['EMA50'] = ta.trend.ema_indicator(data['Close'], window=50)
    macd = ta.trend.MACD(data['Close'])
    data['MACD'] = macd.macd()
    data['MACD_signal'] = macd.macd_signal()
    data['ATR'] = ta.volatility.average_true_range(data['High'], data['Low'], data['Close'], window=14)
    data['BB_upper'] = ta.volatility.bollinger_hband(data['Close'])
    data['BB_lower'] = ta.volatility.bollinger_lband(data['Close'])
    return data

class TradingBot:
    def __init__(self, balance=10000, risk_pct=2):
        self.balance = balance
        self.risk_pct = risk_pct / 100
        self.positions = []
    
    def analyze(self, symbol):
        data = get_data_cached(symbol)
        if data is None or len(data) < 50:
            return None
        
        latest = data.iloc[-1]
        prev = data.iloc[-2]
        
        # Enhanced signal scoring
        bullish = 0
        bearish = 0
        
        # RSI
        if latest['RSI'] < 30: 
            bullish += 2
        elif latest['RSI'] > 70: 
            bearish += 2
        elif latest['RSI'] < 40: 
            bullish += 1
        elif latest['RSI'] > 60: 
            bearish += 1
        
        # EMA trend (strength based on distance)
        ema_diff = (latest['EMA20'] - latest['EMA50']) / latest['Close']
        if ema_diff > 0.005: 
            bullish += 2
        elif ema_diff < -0.005: 
            bearish += 2
        elif ema_diff > 0: 
            bullish += 1
        else: 
            bearish += 1
        
        # MACD
        if latest['MACD'] > latest['MACD_signal'] and prev['MACD'] <= prev['MACD_signal']:
            bullish += 2  # MACD crossover
        elif latest['MACD'] < latest['MACD_signal'] and prev['MACD'] >= prev['MACD_signal']:
            bearish += 2  # MACD crossover
        elif latest['MACD'] > latest['MACD_signal']:
            bullish += 1
        else:
            bearish += 1
        
        # Bollinger Bands
        if latest['Close'] < latest['BB_lower']:
            bullish += 1
        elif latest['Close'] > latest['BB_upper']:
            bearish += 1
        
        # Price momentum (5-period SMA)
        if latest['Close'] > data['Close'].rolling(5).mean().iloc[-1]:
            bullish += 1
        else:
            bearish += 1
        
        confidence = min(abs(bullish - bearish), 6)
        
        if bullish >= 4 and confidence >= 3:
            signal = 'BUY'
        elif bearish >= 4 and confidence >= 3:
            signal = 'SELL'
        else:
            return None
        
        price = latest['Close']
        atr = latest['ATR']
        
        entry = price
        sl_dist = atr * 1.5
        tp_dist = atr * 3
        
        if signal == 'BUY':
            sl = entry - sl_dist
            tp = entry + tp_dist
        else:
            sl = entry + sl_dist
            tp = entry - tp_dist
        
        risk_amount = self.balance * self.risk_pct
        position_size = risk_amount / abs(entry - sl)
        
        return {
            'symbol': symbol,
            'signal': signal,
            'price': price,
            'entry': entry,
            'sl': sl,
            'tp': tp,
            'size': max(position_size, 0.01),  # Minimum size
            'risk': risk_amount,
            'rr': round(tp_dist / sl_dist, 2),
            'confidence': confidence,
            'rsi': round(latest['RSI'], 1),
            'atr': round(atr, 4),
            'data': data.tail(200).reset_index()  # Limit data for plotting
        }
    
    def plot_chart(self, data, symbol, signal=None):
        fig = make_subplots(
            rows=4, cols=1, 
            subplot_titles=(f'{symbol} Price Action', 'RSI (14)', 'MACD', 'Bollinger Bands'),
            vertical_spacing=0.06,
            row_heights=[0.5, 0.2, 0.15, 0.15]
        )
        
        # Candlestick (last 100 candles)
        plot_data = data.tail(100)
        fig.add_trace(go.Candlestick(
            x=plot_data['Datetime'],
            open=plot_data['Open'],
            high=plot_data['High'],
            low=plot_data['Low'],
            close=plot_data['Close'],
            name='Price',
            increasing_line_color='#10b981',
            decreasing_line_color='#ef4444'
        ), row=1, col=1)
        
        # EMAs
        fig.add_trace(go.Scatter(x=plot_data['Datetime'], y=plot_data['EMA20'], 
                                name='EMA20', line=dict(color='orange', width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=plot_data['Datetime'], y=plot_data['EMA50'], 
                                name='EMA50', line=dict(color='blue', width=2)), row=1, col=1)
        
        # Bollinger Bands
        fig.add_trace(go.Scatter(x=plot_data['Datetime'], y=plot_data['BB_upper'], 
                                name='BB Upper', line=dict(color='gray', dash='dash')), row=4, col=1)
        fig.add_trace(go.Scatter(x=plot_data['Datetime'], y=plot_data['BB_lower'], 
                                name='BB Lower', line=dict(color='gray', dash='dash')), row=4, col=1)
        
        # RSI
        fig.add_trace(go.Scatter(x=plot_data['Datetime'], y=plot_data['RSI'], 
                                name='RSI', line=dict(color='purple')), row=2, col=1)
        fig.add_hline(70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(50, line_dash="dot", line_color="gray", row=2, col=1)
        fig.add_hline(30, line_dash="dash", line_color="green", row=2, col=1)
        
        # MACD
        fig.add_trace(go.Scatter(x=plot_data['Datetime'], y=plot_data['MACD'], 
                                name='MACD', line=dict(color='blue')), row=3, col=1)
        fig.add_trace(go.Scatter(x=plot_data['Datetime'], y=plot_data['MACD_signal'], 
                                name='Signal', line=dict(color='red')), row=3, col=1)
        
        fig.update_layout(
            height=800, 
            showlegend=True, 
            title=f"📊 {symbol} - {'🟢 BUY' if signal == 'BUY' else '🔴 SELL'} Signal",
            xaxis_rangeslider_visible=False
        )
        return fig

def main():
    st.markdown('<h1 class="main-header">🤖 AI Forex & Crypto Trading Bot</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("⚙️ Trading Settings")
    balance = st.sidebar.number_input("💰 Account Balance ($)", value=10000.0, min_value=1000.0, step=100.0)
    risk_pct = st.sidebar.slider("🎯 Risk Per Trade (%)", 0.5, 5.0, 2.0, 0.1)
    
    symbols = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCHF=X", "BTC-USD", "ETH-USD", "SOL-USD"]
    selected = st.sidebar.multiselect("📈 Select Instruments", symbols, default=symbols[:4])
    
    refresh = st.sidebar.button("🔄 Refresh Data", help="Force refresh market data")
    
    if refresh:
        st.cache_data.clear()
    
    bot = TradingBot(balance, risk_pct)
    
    # Main dashboard
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="metric-card"><h2>${balance:,.0f}</h2><p>💼 Balance</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><h2>{risk_pct}%</h2><p>⚠️ Risk/Trade</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><h2>{len(selected)}</h2><p>📊 Markets</p></div>', unsafe_allow_html=True)
    
    # Scan button
    if st.button("🚀 SCAN MARKETS NOW", use_container_width=True, type="primary"):
        st.balloons()
        signals = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, symbol in enumerate(selected):
            status_text.text(f'🔍 Analyzing {symbol}... ({i+1}/{len(selected)})')
            signal = bot.analyze(symbol)
            if signal:
                signals.append(signal)
            progress_bar.progress((i + 1) / len(selected))
            time.sleep(0.1)  # Prevent rate limiting
        
        progress_bar.empty()
        status_text.empty()
        
        # Results
        if signals:
            st.success(f"🎉 Found {len(signals)} HIGH-CONFIDENCE SIGNALS!")
            
            for i, signal in enumerate(signals):
                signal_class = f"{'buy-signal' if signal['signal'] == 'BUY' else 'sell-signal'}"
                st.markdown(f"""
                <div class="signal-card {signal_class}">
                    <h3>🚨 {signal['symbol']} - {signal['signal']} ⭐{signal['confidence']}/6</h3>
                    <table style="width:100%; color:white; font-size:14px;">
                        <tr><td><b>Entry:</b></td><td style="float:right;">{signal['entry']:.5f}</td></tr>
                        <tr><td><b>Stop Loss:</b></td><td style="float:right;">{signal['sl']:.5f}</td></tr>
                        <tr><td><b>Take Profit:</b></td><td style="float:right;">{signal['tp']:.5f}</td></tr>
                        <tr><td><b>Position Size:</b></td><td style="float:right;">{signal['size']:.3f}</td></tr>
                        <tr><td><b>Risk Amount:</b></td><td style="float:right;">${signal['risk']:,.0f}</td></tr>
                        <tr><td><b>R:R Ratio:</b></td><td style="float:right;">1:{signal['rr']}</td></tr>
                        <tr><td><b>RSI:</b></td><td style="float:right;">{signal['rsi']}</td></tr>
                        <tr><td><b>ATR:</b></td><td style="float:right;">{signal['atr']}</td></tr>
                    </table>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    fig = bot.plot_chart(signal['data'], signal['symbol'], signal['signal'])
                    st.plotly_chart(fig, use_container_width=True, height=600)
                with col2:
                    st.metric("Current Price", f"{signal['price']:.5f}")
                    st.metric("Risk Amount", f"${signal['risk']:,.0f}")
                    st.metric("R:R Ratio", f"1:{signal['rr']}")
                
                st.divider()
        else:
            st.warning("😴 No high-confidence signals detected. Markets might be ranging. Try again later!")
    
    # Disclaimer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 12px;'>
        ⚠️ <b>DISCLAIMER:</b> This is for educational purposes only. 
        Not financial advice. Always do your own research. Past performance 
        doesn't guarantee future results. Trade at your own risk.
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()