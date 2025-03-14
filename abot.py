#!/usr/bin/env python3
import os
import time
import asyncio
import logging
import json
import ccxt
import pandas as pd
import numpy as np
import requests
import sqlite3
from collections import defaultdict
from pybit.unified_trading import WebSocket, HTTP
from dotenv import load_dotenv
from websockets.exceptions import WebSocketException as WebSocketError

load_dotenv('.env')

# Настройка логирования
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
file_handler = logging.FileHandler('bot.log')
file_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Загрузка настроек из окружения
API_KEY = os.getenv('API_KEY')
API_SECRET = os.getenv('API_SECRET')
if not API_KEY or not API_SECRET:
    logger.critical("Ошибка: API_KEY или API_SECRET не заданы! Проверьте .env файл.")
    exit(1)

LEVERAGE = int(os.getenv('LEVERAGE', '50'))
ORDER_COOLDOWN = int(os.getenv('ORDER_COOLDOWN', '60'))
WHITELIST = os.getenv('WHITELIST', '').strip()
TIMEFRAME = os.getenv('TIMEFRAME', '60')
MIN_VOLUME = float(os.getenv('MIN_VOLUME', '5000'))
ORDER_PERCENTAGE = float(os.getenv('ORDER_PERCENTAGE', '0.02'))
AVERAGING_TRIGGER = float(os.getenv('AVERAGING_TRIGGER', '0.02'))
MAX_AVERAGING = int(os.getenv('MAX_AVERAGING', '3'))
MAX_DRAWDOWN = float(os.getenv('MAX_DRAWDOWN', '-0.10'))
SMA_PERIOD = int(os.getenv('SMA_PERIOD', '300'))
ATR_PERIOD = int(os.getenv('ATR_PERIOD', '14'))
ADX_THRESHOLD = int(os.getenv('ADX_THRESHOLD', '30'))
DB_FILE = "trades.db"

class RateLimiter:
    def __init__(self, calls_per_second):
        self.calls_per_second = calls_per_second
        self.last_call = 0

    async def wait(self):
        elapsed = time.time() - self.last_call
        wait_time = max(1 / self.calls_per_second - elapsed, 0)
        await asyncio.sleep(wait_time)
        self.last_call = time.time()

class TradingBot:
    def __init__(self):
        self.session = None
        self.ws = None
        self.exchange = None
        self.open_positions = {}
        self.last_candle_ts = {}
        self.initial_balance = None
        self.position_lock = asyncio.Lock()
        self.api_limiter = RateLimiter(calls_per_second=5)
        self.symbols = []
        self.setup_connections()
        logger.info("Бот запущен, начинается инициализация...")

    def setup_connections(self):
        try:
            self.session = HTTP(testnet=False, api_key=API_KEY, api_secret=API_SECRET)
            balance = self.session.get_coins_balance(accountType="UNIFIED", coin="USDT")
            usdt_balance = next((float(c["walletBalance"]) for c in balance["result"]["balance"] if c["coin"] == "USDT"), 0.0)
            logger.info(f"API Bybit работает, баланс USDT: {usdt_balance:.2f}")

            self.ws = WebSocket(
                testnet=False,
                channel_type="linear",
                api_key=API_KEY,
                api_secret=API_SECRET
            )

            self.exchange = ccxt.bybit({
                'apiKey': API_KEY,
                'secret': API_SECRET,
                'options': {'defaultType': 'future'}
            })

        except Exception as e:
            logger.critical(f"Ошибка инициализации соединений: {e}")
            raise

    def init_db(self):
        try:
            conn = sqlite3.connect(DB_FILE)
            c = conn.cursor()
            c.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    trade_type TEXT,
                    entry_price REAL,
                    order_size REAL,
                    side TEXT,
                    averaging_count INTEGER,
                    tp REAL,
                    sl REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Ошибка инициализации БД: {e}")

    def log_trade(self, symbol, trade_type, entry_price, order_size, side, averaging_count, tp, sl):
        try:
            conn = sqlite3.connect(DB_FILE)
            c = conn.cursor()
            c.execute('''
                INSERT INTO trades (symbol, trade_type, entry_price, order_size, side, averaging_count, tp, sl)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (symbol, trade_type, entry_price, order_size, side, averaging_count, tp, sl))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Ошибка записи в БД: {e}")

    async def fetch_candles(self, symbol, timeframe=TIMEFRAME, limit=SMA_PERIOD + 10):
        await self.api_limiter.wait()
        for attempt in range(3):
            try:
                logger.debug(f"Запрос {limit} свечей для {symbol}... Попытка {attempt+1}")
                response = self.session.get_kline(
                    category="linear",
                    symbol=symbol,
                    interval=timeframe,
                    limit=limit
                )
                data = response["result"]["list"]
                if not data:
                    logger.warning(f"Пустой ответ от API для {symbol}. Попытка {attempt+1}/3")
                    await asyncio.sleep(1)
                    continue

                column_names = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'][:len(data[0])]
                df = pd.DataFrame(data, columns=column_names)
                df['symbol'] = symbol

                df['timestamp'] = pd.to_datetime(df['timestamp'].astype('int64'), unit='ms')
                numeric_cols = ['open', 'high', 'low', 'close', 'volume']
                df[numeric_cols] = df[numeric_cols].astype(float)

                df = df.drop_duplicates(subset=['timestamp'])
                if len(df) < SMA_PERIOD:
                    logger.warning(f"Недостаточно данных для SMA для {symbol}: {len(df)} < {SMA_PERIOD}")
                    return None

                return df

            except Exception as e:
                logger.error(f"Ошибка получения свечей {symbol}: {str(e)}")
                return None

    def get_orderbook_clusters(self, symbol, window=10):
        try:
            orderbook = self.exchange.fetch_order_book(symbol)
            bids = orderbook['bids'][:window]
            asks = orderbook['asks'][:window]
            
            total_volume = sum([amt for _, amt in bids + asks])
            if total_volume < MIN_VOLUME * 0.1:
                logger.warning(f"Слишком низкий объем для {symbol} ({total_volume:.2f} < {MIN_VOLUME*0.1:.2f})")
                return []

            cluster_levels = defaultdict(float)
            for price, amount in bids + asks:
                if amount < MIN_VOLUME:
                    continue
                rounded_price = round(price, 1)
                cluster_levels[rounded_price] += amount
            return sorted(cluster_levels.items(), key=lambda x: x[1], reverse=True)[:3]
            
        except Exception as e:
            logger.error(f"Ошибка анализа стакана {symbol}: {e}")
            return []

    def calculate_sma(self, data, period=SMA_PERIOD):
        return data['close'].rolling(window=period).mean()

    def calculate_adx(self, data, period=14):
        high, low, close = data['high'], data['low'], data['close']
        tr = np.maximum(high - low, np.abs(high - close.shift()), np.abs(low - close.shift()))
        atr = tr.rolling(window=period).mean().fillna(0)
        plus_dm = np.where(high - high.shift() > low.shift() - low, high - high.shift(), 0)
        minus_dm = np.where(low.shift() - low > high - high.shift(), low.shift() - low, 0)
        plus_di = 100 * (pd.Series(plus_dm).rolling(window=period).mean().fillna(0) / atr)
        minus_di = 100 * (pd.Series(minus_dm).rolling(window=period).mean().fillna(0) / atr)
        dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
        return dx.rolling(window=period).mean().fillna(0)

    def calculate_di(self, data, period=14):
        high, low, close = data['high'], data['low'], data['close']
        tr = np.maximum(high - low, np.abs(high - close.shift()), np.abs(low - close.shift()))
        atr = tr.rolling(window=period).mean()
        plus_dm = np.where(high - high.shift() > low.shift() - low, high - high.shift(), 0)
        minus_dm = np.where(low.shift() - low > high - high.shift(), low.shift() - low, 0)
        plus_di = 100 * (pd.Series(plus_dm).rolling(window=period).mean() / atr)
        minus_di = 100 * (pd.Series(minus_dm).rolling(window=period).mean() / atr)
        return plus_di.iloc[-1], minus_di.iloc[-1]

    def calculate_atr(self, data, period=ATR_PERIOD):
        tr = pd.concat([
            data['high'] - data['low'],
            abs(data['high'] - data['close'].shift()),
            abs(data['low'] - data['close'].shift())
        ], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    def determine_trend(self, df):
        try:
            sma = self.calculate_sma(df)
            adx = self.calculate_adx(df)
            current_price = df['close'].iloc[-1]
            last_sma = sma.iloc[-1]
            last_adx = adx.iloc[-1]
            plus_di, minus_di = self.calculate_di(df)

            if last_adx > ADX_THRESHOLD:
                if current_price > last_sma and plus_di > minus_di:
                    return "uptrend"
                elif current_price < last_sma and minus_di > plus_di:
                    return "downtrend"
            return "flat"
        except Exception as e:
            logger.error(f"Ошибка определения тренда: {e}")
            return "flat"

    async def execute_order(self, symbol, price, side, qty, order_type="Limit"):
        async with self.position_lock:
            await self.api_limiter.wait()
            try:
                market_info = self.session.get_instruments_info(category="linear", symbol=symbol)
                lot_size_filter = market_info["result"]["list"][0]["lotSizeFilter"]
                qty_step = float(lot_size_filter["qtyStep"])
                adjusted_qty = round(qty / qty_step) * qty_step

                order = self.session.place_order(
                    category="linear",
                    symbol=symbol,
                    side=side,
                    orderType=order_type,
                    qty=adjusted_qty,
                    price=round(price, 4) if price is not None else None,
                    timeInForce="GTC",
                    leverage=LEVERAGE
                )
                logger.info(f"Ордер {side} {symbol} размещен! ID: {order['result']['orderId']}")
                return adjusted_qty
            except Exception as e:
                logger.error(f"Ошибка размещения ордера {symbol}: {e}")
                return False

    async def close_position(self, symbol):
        async with self.position_lock:
            try:
                pos = self.open_positions[symbol]
                close_side = "Sell" if pos["side"] == "Buy" else "Buy"
                await self.execute_order(symbol, None, close_side, pos["order_size"], "Market")
                del self.open_positions[symbol]
                logger.info(f"Позиция {symbol} закрыта")
            except Exception as e:
                logger.error(f"Ошибка закрытия {symbol}: {e}")

    async def check_drawdown(self):
        while True:
            await asyncio.sleep(60)
            try:
                balance = self.session.get_coins_balance(accountType="UNIFIED", coin="USDT")
                current_balance = next((float(c["walletBalance"]) for c in balance["result"]["balance"] if c["coin"] == "USDT"), 0.0)
                if self.initial_balance is None:
                    self.initial_balance = current_balance
                    continue
                drawdown = (current_balance - self.initial_balance) / self.initial_balance
                if drawdown < MAX_DRAWDOWN:
                    logger.critical(f"Просадка {drawdown:.2%}! Закрытие позиций...")
                    for sym in list(self.open_positions.keys()):
                        await self.close_position(sym)
                    return
            except Exception as e:
                logger.error(f"Ошибка мониторинга баланса: {e}")

    async def keep_alive(self):
        while True:
            await asyncio.sleep(300)
            try:
                if not self.ws.is_connected():
                    await self.ws._connect()
                self.session.get_orderbook(category="linear", symbol="BTCUSDT")
            except Exception as e:
                logger.error(f"Keep-alive failed: {e}")

    async def handle_message(self, message):
        try:
            if 'topic' not in message:
                return

            topic = message['topic']
            if 'kline' in topic:
                symbol = topic.split('.')[-1]
                data = message['data'][0]
                candle_time = pd.to_datetime(data['start'], unit='ms')
                
                async with self.position_lock:
                    self.last_candle_ts[symbol] = candle_time

        except Exception as e:
            logger.error(f"Ошибка обработки WebSocket: {e}")

    async def ws_listener(self):
        while True:
            try:
                if not self.ws.is_connected():
                    await asyncio.sleep(1)
                    continue

                message = await self.ws.recv()
                await self.handle_message(message)

            except WebSocketError as e:
                logger.error(f"WebSocket error: {e}")
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"Ошибка слушателя: {e}")
                await asyncio.sleep(1)

    def fetch_symbols(self):
        try:
            if WHITELIST:
                self.symbols = [s.strip() for s in WHITELIST.split(',') if s.strip()]
            else:
                response = requests.get("https://api.bybit.com/v5/market/instruments-info?category=linear")
                data = response.json()
                self.symbols = [
                    s["symbol"] for s in data.get("result", {}).get("list", [])
                    if s.get("quoteCoin") == "USDT" and 
                    float(s.get("lotSizeFilter", {}).get("maxOrderQty", 0)) > MIN_VOLUME
                ]

            logger.info(f"Список торговых пар: {self.symbols}")
            return True
        except Exception as e:
            logger.critical(f"Ошибка получения символов: {e}")
            return False

    async def process_candle(self, symbol):
        await self.api_limiter.wait()
        df = await self.fetch_candles(symbol)
        if df is None or len(df) < SMA_PERIOD:
            return

        last_ts = df['timestamp'].iloc[-1]
        if symbol in self.last_candle_ts and self.last_candle_ts[symbol] == last_ts:
            return
        self.last_candle_ts[symbol] = last_ts

        trend = self.determine_trend(df)
        atr_series = self.calculate_atr(df)
        current_price = df['close'].iloc[-1]

        if trend == "flat":
            return

        balance = self.session.get_coins_balance(accountType="UNIFIED", coin="USDT")
        usdt_balance = next((float(c["walletBalance"]) for c in balance["result"]["balance"] if c["coin"] == "USDT"), 0.0)
        order_usdt = usdt_balance * ORDER_PERCENTAGE

        best_support, best_resistance = self.get_historical_levels(df)
        orderbook_clusters = self.get_orderbook_clusters(symbol)

        key_level = None
        side = "Buy" if trend == "uptrend" else "Sell"
        key_level = self.find_matching_level(
            best_support if trend == "uptrend" else best_resistance,
            orderbook_clusters
        )

        if key_level is None or abs(current_price - key_level) / key_level > 0.005:
            return

        raw_qty = order_usdt / key_level
        placed_qty = await self.execute_order(symbol, key_level, side, raw_qty)
        
        if placed_qty:
            tp, sl = self.calculate_tp_sl(key_level, side, atr_series.iloc[-1])
            self.open_positions[symbol] = {
                "entry_price": key_level,
                "order_size": placed_qty,
                "side": side,
                "tp": tp,
                "sl": sl,
                "averaging_count": 0
            }
            self.log_trade(symbol, "initial", key_level, placed_qty, side, 0, tp, sl)

    def get_historical_levels(self, df, period=50):
        recent_data = df[-period:]
        return recent_data["low"].max(), recent_data["high"].min()

    def find_matching_level(self, hist_level, orderbook_clusters, tolerance=0.005):
        if hist_level is None:
            return None
        for price, _ in orderbook_clusters:
            if abs(hist_level - price) / hist_level < tolerance:
                return price
        return None

    def calculate_tp_sl(self, entry_price, side, atr):
        risk_factor = 1.5
        if side == "Buy":
            return entry_price + risk_factor * atr, entry_price - atr
        else:
            return entry_price - risk_factor * atr, entry_price + atr

    async def main_loop(self):
        self.init_db()
        if not self.fetch_symbols():
            logger.error("Нет доступных торговых пар")
            return

        tasks = [
            asyncio.create_task(self.ws_listener()),
            asyncio.create_task(self.keep_alive()),
            asyncio.create_task(self.check_drawdown())
        ]

        while True:
            for symbol in self.symbols:
                await self.process_candle(symbol)
            await asyncio.sleep(int(TIMEFRAME))

    async def shutdown(self):
        logger.info("Завершение работы...")
        try:
            for symbol in list(self.open_positions.keys()):
                await self.close_position(symbol)
            if self.ws.is_connected():
                await self.ws.close()
        except Exception as e:
            logger.error(f"Ошибка завершения: {e}")

if __name__ == '__main__':
    bot = TradingBot()
    try:
        asyncio.run(bot.main_loop())
    except KeyboardInterrupt:
        logger.info("Завершение по сигналу пользователя")
        asyncio.run(bot.shutdown())
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
        asyncio.run(bot.shutdown())