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

# Импортируем исключение для обработки ошибок WebSocket
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
logger.info("Бот запущен, начинается инициализация...")

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

# Подключение к API Bybit через HTTP
logger.info("Подключение к API Bybit...")
try:
    session = HTTP(testnet=False, api_key=API_KEY, api_secret=API_SECRET)
    balance = session.get_coins_balance(accountType="UNIFIED", coin="USDT")
    usdt_balance = next((float(c["walletBalance"]) for c in balance["result"]["balance"] if c["coin"] == "USDT"), 0.0)
    logger.info(f"API Bybit работает, баланс USDT: {usdt_balance:.2f}")
except Exception as e:
    logger.critical(f"Ошибка подключения к API: {e}")
    exit(1)

# Создаём WebSocket для получения обновлений
ws = WebSocket(
    testnet=False,
    channel_type="linear",
    api_key=API_KEY,
    api_secret=API_SECRET
)

# Создаём объект для вызова методов биржи через ccxt (используется для анализа стакана)
exchange = ccxt.bybit({
    'apiKey': API_KEY,
    'secret': API_SECRET,
    'options': {'defaultType': 'future'}
})

retries = 10  # Количество попыток запроса

last_candle_ts = {}
open_positions = {}
position_lock = asyncio.Lock()
initial_balance = None

class RateLimiter:
    def __init__(self, calls_per_second):
        self.calls_per_second = calls_per_second
        self.last_call = 0

    async def wait(self):
        elapsed = time.time() - self.last_call
        wait_time = max(1 / self.calls_per_second - elapsed, 0)
        await asyncio.sleep(wait_time)
        self.last_call = time.time()

api_limiter = RateLimiter(calls_per_second=5)

def init_db():
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

def log_trade(symbol, trade_type, entry_price, order_size, side, averaging_count, tp, sl):
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

async def fetch_candles(symbol, timeframe=TIMEFRAME, limit=SMA_PERIOD + 10):
    await api_limiter.wait()
    for attempt in range(3):
        try:
            logger.debug(f"Запрос {limit} свечей для {symbol}... Попытка {attempt+1}")
            response = session.get_kline(
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

            if not all(isinstance(candle, list) for candle in data):
                logger.error(f"Некорректный формат данных для {symbol}")
                return None

            num_columns = len(data[0])
            column_names = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'][:num_columns]
            df = pd.DataFrame(data, columns=column_names)
            df['symbol'] = symbol

            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'].astype('int64'), unit='ms')
                numeric_cols = ['open', 'high', 'low', 'close', 'volume']
                df[numeric_cols] = df[numeric_cols].astype(float)
            except Exception as e:
                logger.error(f"Ошибка конвертации типов для {symbol}: {str(e)}")
                return None

            df = df.drop_duplicates(subset=['timestamp'])
            logger.debug(f"Уникальных свечей для {symbol} после фильтрации: {len(df)}")
            if len(df) < SMA_PERIOD:
                logger.warning(f"Недостаточно данных для SMA для {symbol}: {len(df)} < {SMA_PERIOD}")
                return None

            return df

        except KeyError as e:
            logger.error(f"Ключ {e} не найден в ответе API для {symbol}")
            return None
        except Exception as e:
            logger.error(f"Критическая ошибка для {symbol}: {str(e)}")
            return None

def get_orderbook_clusters(symbol, window=10):
    try:
        orderbook = exchange.fetch_order_book(symbol)
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

def get_historical_levels(df, period=50):
    recent_data = df[-period:]
    support_level = recent_data["low"].max()
    resistance_level = recent_data["high"].min()
    return support_level, resistance_level

def find_liquidity_zones(symbol, df):
    support, resistance = get_historical_levels(df, period=50)
    cluster_levels = get_orderbook_clusters(symbol)
    liquidity_zоны = {"support": support, "resistance": resistance}
    for price, _ in cluster_levels:
        if abs(price - support) <= 0.1 * support:
            liquidity_zоны["support"] = price
        if abs(price - resistance) <= 0.1 * resistance:
            liquidity_zоны["resistance"] = price
    return liquidity_zоны

def find_support_resistance(df, window=20, tolerance=0.005):
    supports = []
    resistances = []
    for i in range(window, len(df)-window):
        window_df = df.iloc[i-window:i+window+1]
        current_low = df['low'].iloc[i]
        current_high = df['high'].iloc[i]
        if current_low == window_df['low'].min():
            supports.append(current_low)
        if current_high == window_df['high'].max():
            resistances.append(current_high)

    support_clusters = []
    for level in sorted(supports):
        added = False
        for cluster in support_clusters:
            if abs(level - cluster[0]) / cluster[0] < tolerance:
                cluster[1] += 1
                cluster[0] = (cluster[0] * (cluster[1]-1) + level) / cluster[1]
                added = True
                break
        if not added:
            support_clusters.append([level, 1])

    resistance_clusters = []
    for level in sorted(resistances, reverse=True):
        added = False
        for cluster in resistance_clusters:
            if abs(level - cluster[0]) / cluster[0] < tolerance:
                cluster[1] += 1
                cluster[0] = (cluster[0] * (cluster[1]-1) + level) / cluster[1]
                added = True
                break
        if not added:
            resistance_clusters.append([level, 1])

    support_clusters.sort(key=lambda x: x[1], reverse=True)
    resistance_clusters.sort(key=lambda x: x[1], reverse=True)
    best_support = support_clusters[0][0] if support_clusters else None
    best_resistance = resistance_clusters[0][0] if resistance_clusters else None
    return best_support, best_resistance, support_clusters, resistance_clusters

def calculate_sma(data, period=SMA_PERIOD):
    return data['close'].rolling(window=period).mean()

def calculate_adx(data, period=14):
    high, low, close = data['high'], data['low'], data['close']
    tr = np.maximum(high - low, np.abs(high - close.shift()), np.abs(low - close.shift()))
    atr = tr.rolling(window=period).mean().fillna(0)
    plus_dm = np.where(high - high.shift() > low.shift() - low, high - high.shift(), 0)
    minus_dm = np.where(low.shift() - low > high - high.shift(), low.shift() - low, 0)
    plus_di = 100 * (pd.Series(plus_dm).rolling(window=period).mean().fillna(0) / atr)  # Добавлена закрывающая скобка
    minus_di = 100 * (pd.Series(minus_dm).rolling(window=period).mean().fillna(0) / atr)  # Исправлено
    dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    return dx.rolling(window=period).mean().fillna(0)

def calculate_di(data, period=14):
    high, low, close = data['high'], data['low'], data['close']
    tr = np.maximum(high - low, np.abs(high - close.shift()), np.abs(low - close.shift()))
    atr = tr.rolling(window=period).mean()
    plus_dm = np.where(high - high.shift() > low.shift() - low, high - high.shift(), 0)
    minus_dm = np.where(low.shift() - low > high - high.shift(), low.shift() - low, 0)
    plus_di = 100 * (pd.Series(plus_dm).rolling(window=period).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm).rolling(window=period).mean() / atr)
    return plus_di.iloc[-1], minus_di.iloc[-1]

def calculate_atr(data, period=ATR_PERIOD):
    tr = pd.concat([
        data['high'] - data['low'],
        abs(data['high'] - data['close'].shift()),
        abs(data['low'] - data['close'].shift())
    ], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def calculate_rsi(data, period=14):
    delta = data['close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def determine_trend(df):
    try:
        sma = calculate_sma(df, period=SMA_PERIOD)
        adx = calculate_adx(df, period=14)
        rsi = calculate_rsi(df, period=14)
        current_price = df['close'].iloc[-1]
        last_sma = sma.iloc[-1]
        last_adx = adx.iloc[-1]
        last_rsi = rsi.iloc[-1]
        plus_di, minus_di = calculate_di(df, period=14)

        logger.info(f"[{df['symbol'].iloc[0]}] Индикаторы: SMA={last_sma:.2f}, ADX={last_adx:.2f}, RSI={last_rsi:.2f}, +DI={plus_di:.2f}, -DI={minus_di:.2f}, Цена={current_price:.2f}")

        if sma.isnull().any() or (sma == 0).any():
            logger.error(f"Недостаточно данных для SMA ({len(df)} < {SMA_PERIOD})")
            return "flat"

        if last_adx > ADX_THRESHOLD:
            if current_price > last_sma and last_rsi > 55 and plus_di > minus_di:
                trend = "uptrend"
            elif current_price < last_sma and last_rsi < 45 and minus_di > plus_di:
                trend = "downtrend"
            else:
                trend = "flat"
        else:
            trend = "flat"

        logger.info(f"Определен тренд: {trend}")
        return trend

    except Exception as e:
        logger.error(f"Ошибка определения тренда: {e}")
        return "flat"

def calculate_tp_sl(entry_price, side, atr):
    risk_factor = 1.5
    if side == "Buy":
        tp = entry_price + risk_factor * atr
        sl = entry_price - atr
    else:
        tp = entry_price - risk_factor * atr
        sl = entry_price + atr
    return round(tp, 4), round(sl, 4)

def find_matching_level(hist_level, orderbook_clusters, tolerance=0.005):
    if hist_level is None:
        return None
    for price, _ in orderbook_clusters:
        if abs(hist_level - price) / hist_level < tolerance:
            return price
    return None

async def execute_order(symbol, price, side, qty, order_type="Limit"):
    async with position_lock:
        await api_limiter.wait()
        try:
            market_info = session.get_instruments_info(category="linear", symbol=symbol)
            lot_size_filter = market_info["result"]["list"][0]["lotSizeFilter"]
            qty_step = float(lot_size_filter["qtyStep"])
            adjusted_qty = round(qty / qty_step) * qty_step

            order = session.place_order(
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

async def close_position(symbol):
    async with position_lock:
        try:
            pos = open_positions[symbol]
            close_side = "Sell" if pos["side"] == "Buy" else "Buy"
            await execute_order(symbol, None, close_side, pos["order_size"], "Market")
            del open_positions[symbol]
            logger.info(f"Позиция {symbol} закрыта")
        except Exception as e:
            logger.error(f"Ошибка закрытия {symbol}: {e}")

async def process_candle(symbol):
    await api_limiter.wait()
    df = await fetch_candles(symbol)
    if df is None or len(df) < SMA_PERIOD:
        return

    last_ts = df['timestamp'].iloc[-1]
    if symbol in last_candle_ts and last_candle_ts[symbol] == last_ts:
        return
    last_candle_ts[symbol] = last_ts

    trend = determine_trend(df)
    atr_series = calculate_atr(df, period=14)
    if atr_series is None or atr_series.empty:
        return
    atr = atr_series.iloc[-1]
    current_price = df['close'].iloc[-1]

    logger.info(f"[{symbol}] Проверка условий входа: Тренд={trend}, Цена={current_price:.2f}, ATR={atr:.2f}")

    if trend == "flat":
        logger.info(f"[{symbol}] Тренд 'flat', пропуск")
        return

    balance = session.get_coins_balance(accountType="UNIFIED", coin="USDT")
    usdt_balance = next((float(c["walletBalance"]) for c in balance["result"]["balance"] if c["coin"] == "USDT"), 0.0)
    order_usdt = usdt_balance * ORDER_PERCENTAGE
    logger.info(f"[{symbol}] Баланс: {usdt_balance:.2f} USDT | Размер ордера: {order_usdt:.2f} USDT")

    best_support, best_resistance, support_clusters, resistance_clusters = find_support_resistance(df, window=20, tolerance=0.005)
    orderbook_clusters = get_orderbook_clusters(symbol)

    logger.info(f"[{symbol}] Уровни: Поддержка={best_support:.2f}, Сопротивление={best_resistance:.2f}")
    logger.debug(f"[{symbol}] Кластеры стакана: {orderbook_clusters}")

    key_level = None
    side = None
    if trend == "uptrend":
        key_level = find_matching_level(best_support, orderbook_clusters, tolerance=0.005)
        side = "Buy"
    elif trend == "downtrend":
        key_level = find_matching_level(best_resistance, orderbook_clusters, tolerance=0.005)
        side = "Sell"

    logger.info(f"[{symbol}] Ключевой уровень: {key_level}")

    if key_level is None or abs(current_price - key_level) / key_level > 0.005:
        logger.info(f"[{symbol}] Уровень {key_level} не соответствует условиям входа")
        return

    raw_qty = order_usdt / key_level
    placed_qty = await execute_order(symbol, key_level, side, raw_qty)
    
    if placed_qty:
        tp, sl = calculate_tp_sl(key_level, side, atr)
        open_positions[symbol] = {
            "entry_price": key_level,
            "order_size": placed_qty,
            "last_order_size": placed_qty,
            "averaging_count": 0,
            "side": side,
            "tp": tp,
            "sl": sl
        }
        log_trade(symbol, "initial", key_level, placed_qty, side, 0, tp, sl)
    else:
        pos = open_positions.get(symbol)
        if not pos:
            return
        if (pos["side"] == "Buy" and current_price >= pos["tp"]) or \
           (pos["side"] == "Sell" and current_price <= pos["tp"]):
            await close_position(symbol)
        elif (pos["side"] == "Buy" and current_price <= pos["sl"]) or \
             (pos["side"] == "Sell" and current_price >= pos["sl"]):
            await close_position(symbol)
        elif pos["averaging_count"] < MAX_AVERAGING:
            if (pos["side"] == "Buy" and current_price < pos["entry_price"] - AVERAGING_TRIGGER * atr) or \
               (pos["side"] == "Sell" and current_price > pos["entry_price"] + AVERAGING_TRIGGER * atr):
                additional_qty = pos["order_size"] * 0.5
                placed_qty = await execute_order(symbol, current_price, pos["side"], additional_qty)
                if placed_qty:
                    total_qty = pos["order_size"] + placed_qty
                    new_entry_price = (pos["entry_price"] * pos["order_size"] + current_price * placed_qty) / total_qty
                    pos["entry_price"] = new_entry_price
                    pos["order_size"] = total_qty
                    pos["averaging_count"] += 1
                    tp, sl = calculate_tp_sl(new_entry_price, pos["side"], atr)
                    pos["tp"] = tp
                    pos["sl"] = sl
                    log_trade(symbol, "averaging", new_entry_price, placed_qty, pos["side"], pos["averaging_count"], tp, sl)

def check_drawdown_sync(current_balance, initial_balance):
    return (current_balance - initial_balance) / initial_balance

async def check_drawdown():
    global initial_balance
    while True:
        await asyncio.sleep(60)
        try:
            balance = session.get_coins_balance(accountType="UNIFIED", coin="USDT")
            current_balance = next((float(c["walletBalance"]) for c in balance["result"]["balance"] if c["coin"] == "USDT"), 0.0)
            if initial_balance is None:
                initial_balance = current_balance
                continue
            drawdown = check_drawdown_sync(current_balance, initial_balance)
            logger.debug(f"Баланс: {current_balance}, Начальный баланс: {initial_balance}, Просадка: {drawdown:.2%}")

            if drawdown < MAX_DRAWDOWN:
                logger.critical(f"Просадка {drawdown:.2%}! Закрытие позиций...")
                for sym in list(open_positions.keys()):
                    await close_position(sym)
                logger.info("Все позиции закрыты. Завершение работы.")
                return
        except Exception as e:
            logger.error(f"Ошибка мониторинга баланса: {e}")

async def keep_alive():
    while True:
        await asyncio.sleep(300)
        try:
            if not ws.is_connected():
                logger.warning("Переподключение WebSocket...")
                await ws._connect()
            # Пинг-запрос к API для поддержания активности соединения
            session.get_orderbook(category="linear", symbol="BTCUSDT")
        except Exception as e:
            logger.error(f"Keep-alive failed: {e}")

async def handle_message(message):
    """Обработчик сообщений WebSocket"""
    try:
        if 'topic' not in message:
            return

        topic = message['topic']
        if 'kline' in topic:
            symbol = topic.split('.')[-1]
            data = message['data'][0]
            candle_time = pd.to_datetime(data['start'], unit='ms')
            
            # Обновляем последнее время свечи
            async with position_lock:
                last_candle_ts[symbol] = candle_time

    except Exception as e:
        logger.error(f"Ошибка обработки WebSocket: {e}")

async def ws_listener():
    """Слушатель WebSocket"""
    while True:
        try:
            if not ws.is_connected():
                await asyncio.sleep(1)
                continue

            message = await ws.recv()
            await handle_message(message)

        except WebSocketError as e:
            logger.error(f"WebSocket error: {e}")
            await asyncio.sleep(5)
        except Exception as e:
            logger.error(f"Ошибка слушателя: {e}")
            await asyncio.sleep(1)

def fetch_symbols():
    logger.info("Загрузка списка торговых пар...")
    symbols = []
    try:
        if WHITELIST:
            symbols = [s.strip() for s in WHITELIST.split(',') if isinstance(s.strip(), str) and s.strip()]
        else:
            response = requests.get("https://api.bybit.com/v5/market/instruments-info?category=linear")
            data = response.json()
            symbols = [
                s["symbol"] for s in data.get("result", {}).get("list", [])
                if isinstance(s, dict) and s.get("quoteCoin") == "USDT" and float(s.get("lotSizeFilter", {}).get("maxOrderQty", 0)) > MIN_VOLUME
            ]

        if not isinstance(symbols, list):  # Дополнительная проверка
            logger.critical("Ошибка: список торговых пар не является списком!")
            return []

        logger.info(f"Список торговых пар: {symbols}")
        return symbols

    except Exception as e:
        logger.critical(f"Ошибка получения символов: {e}")
        return []

    except Exception as e:
        logger.critical(f"Ошибка получения символов: {e}")
        return []

async def main():
    init_db()
    symbols = fetch_symbols()

    if not symbols:
        logger.error("Нет доступных торговых пар. Завершение работы.")
        return

    # Подписка на WebSocket (актуальный формат)
    for symbol in symbols:
        if not isinstance(symbol, str):  # Проверяем, что symbol - строка
            logger.error(f"Некорректный символ: {symbol}")
            continue
        topic = f"kline.{TIMEFRAME}.{symbol}"
        ws.subscribe(topic, callback=handle_message)
        logger.info(f"Подписка на {topic}")

    # Запуск задач
    tasks = [
        asyncio.create_task(ws_listener()),
        asyncio.create_task(keep_alive()),
        asyncio.create_task(check_drawdown())
    ]

    # Основной цикл обработки
    while True:
        for symbol in symbols:
            await process_candle(symbol)
        await asyncio.sleep(int(TIMEFRAME))

    await asyncio.gather(*tasks, return_exceptions=True)



if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Завершение работы по KeyboardInterrupt.")

