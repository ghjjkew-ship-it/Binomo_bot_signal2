import os
import json
import sqlite3
import io
from datetime import datetime

import aiohttp
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from aiogram import Bot, Dispatcher, types
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup
from aiogram.utils import executor
from sklearn.linear_model import SGDClassifier

# ===================== НАСТРОЙКИ =====================
TG_TOKEN = "8454094681:AAE6_6BSaEkQZabrxjEcDhIgMQSBbFMPqRI"  # <-- замените на ваш токен
TWELVEDATA_API_KEY = "74c58d1151144bb990851b622ba809b2"  # <-- замените на ваш ключ

DB_FILE = "bot_data.sqlite"

PAIRS = ["AUD/CAD", "EUR/USD", "USD/JPY"]
TIMEFRAMES = ["1min", "5min", "15min"]

# ===================== ИНИЦИАЛИЗАЦИЯ =====================
bot = Bot(token=TG_TOKEN)
dp = Dispatcher(bot)

conn = sqlite3.connect(DB_FILE, check_same_thread=False)
cur = conn.cursor()

# Создаем таблицы пользователей и сигналов, если их нет
cur.execute("""
CREATE TABLE IF NOT EXISTS users (
    user_id INTEGER PRIMARY KEY,
    username TEXT,
    pair TEXT DEFAULT 'AUD/CAD',
    timeframe TEXT DEFAULT '1min',
    start_ts TEXT
)
""")
cur.execute("""
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT,
    chat_id INTEGER,
    message_id INTEGER,
    pair TEXT,
    timeframe TEXT,
    features TEXT,
    predicted INTEGER,
    label INTEGER
)
""")
conn.commit()

# ===================== ФУНКЦИИ =====================

def model_filename(pair, timeframe):
    safe_pair = pair.replace("/", "_")
    return f"model_{safe_pair}_{timeframe}.joblib"

def load_model(pair, timeframe):
    fname = model_filename(pair, timeframe)
    if os.path.exists(fname):
        try:
            return joblib.load(fname)
        except:
            pass
    model = SGDClassifier(loss="log_loss")
    model.partial_fit(np.zeros((1,5)), [0], classes=[0,1])
    return model

def save_model(model, pair, timeframe):
    fname = model_filename(pair, timeframe)
    joblib.dump(model, fname)

def get_user_settings(user_id):
    cur.execute("SELECT pair, timeframe FROM users WHERE user_id = ?", (user_id,))
    row = cur.fetchone()
    if row:
        return row[0], row[1]
    # Если пользователя нет — создаём с дефолтами
    cur.execute("INSERT OR IGNORE INTO users (user_id, username, start_ts) VALUES (?, ?, ?)",
                (user_id, "unknown", datetime.utcnow().isoformat()))
    conn.commit()
    return "AUD/CAD", "1min"

def set_user_pair(user_id, pair):
    cur.execute("UPDATE users SET pair = ? WHERE user_id = ?", (pair, user_id))
    conn.commit()

def set_user_timeframe(user_id, timeframe):
    cur.execute("UPDATE users SET timeframe = ? WHERE user_id = ?", (timeframe, user_id))
    conn.commit()

def compute_features_from_closes(closes: pd.Series):
    if len(closes) < 16:
        closes = pd.Series(list(closes) + [closes.iloc[-1]] * (16 - len(closes)))
    last3 = closes.values[-3:]
    sma = closes.rolling(14).mean().iloc[-1]
    sma_prev = closes.rolling(14).mean().iloc[-2]
    sma_diff = 0.0 if pd.isna(sma) or pd.isna(sma_prev) else (sma - sma_prev)
    delta = closes.diff().fillna(0)
    up = delta.clip(lower=0).rolling(14).mean().iloc[-1]
    down = -delta.clip(upper=0).rolling(14).mean().iloc[-1]
    rsi = 50.0
    if (up + down) != 0:
        rs = up / (down if down != 0 else 1e-9)
        rsi = 100 - (100 / (1 + rs))
    denom = last3[-1] if last3[-1] != 0 else 1
    norm_last3 = last3 / denom
    feats = np.concatenate([norm_last3, [sma_diff, rsi]])
    return feats.reshape(1, -1)

async def fetch_twelvedata(pair="AUD/CAD", interval="1min", outputsize=100):
    url = (
        "https://api.twelvedata.com/time_series"
        f"?symbol={pair}"
        f"&interval={interval}"
        f"&outputsize={outputsize}"
        f"&format=JSON"
        f"&apikey={TWELVEDATA_API_KEY}"
    )
    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=15) as resp:
            data = await resp.json()
    if "values" in data:
        df = pd.DataFrame(data["values"])
        df = df.sort_values("datetime")
        df["close"] = df["close"].astype(float)
        return df["close"].reset_index(drop=True)
    else:
        raise RuntimeError(f"TwelveData error: {data}")

async def get_closes(pair="AUD/CAD", interval="1min"):
    try:
        return await fetch_twelvedata(pair, interval)
    except Exception as e:
        raise RuntimeError("Ошибка получения данных: " + str(e))

def decide_from_features(feats, model):
    rsi = feats[0, -1]
    if rsi < 30:
        return 1  # BUY
    if rsi > 70:
        return 0  # SELL
    pred = int(model.predict(feats)[0])
    return pred

def signal_keyboard():
    kb = InlineKeyboardMarkup(row_width=2)
    kb.add(
        InlineKeyboardButton("Верный ✅", callback_data="fb|1"),
        InlineKeyboardButton("Неверный ❌", callback_data="fb|0")
    )
    return kb

def settings_keyboard(user_id):
    kb = InlineKeyboardMarkup(row_width=1)
    # Пары
    for pair in PAIRS:
        kb.insert(InlineKeyboardButton(f"Пара: {pair}", callback_data=f"setpair|{pair}|{user_id}"))
    # Таймфреймы
    for tf in TIMEFRAMES:
        kb.insert(InlineKeyboardButton(f"Таймфрейм: {tf}", callback_data=f"settf|{tf}|{user_id}"))
    kb.add(InlineKeyboardButton("Получить новый сигнал /новый", callback_data=f"newsignal|{user_id}"))
    kb.add(InlineKeyboardButton("Показать статистику /статистика", callback_data=f"stats|{user_id}"))
    return kb

async def send_signal(chat_id, dir_text, feats, pair, timeframe):
    text = (f"📢 <b>Сигнал: {pair} ({timeframe})</b>\n"
            f"Направление: <b>{dir_text}</b>\n"
            f"Время (UTC): {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            "Нажми: Верный / Неверный для обучения бота")
    msg = await bot.send_message(chat_id, text, parse_mode="HTML", reply_markup=signal_keyboard())
    feats_json = json.dumps(feats.tolist())
    predicted = 1 if dir_text == "BUY" else 0
    cur.execute(
        "INSERT INTO signals (ts, chat_id, message_id, pair, timeframe, features, predicted, label) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (datetime.utcnow().isoformat(), chat_id, msg.message_id, pair, timeframe, feats_json, predicted, None),
    )
    conn.commit()
    return msg.message_id

def plot_accuracy_chart():
    cur.execute("SELECT label, predicted FROM signals WHERE label IS NOT NULL")
    data = cur.fetchall()
    if not data:
        return None
    labels = [d[0] for d in data]
    preds = [d[1] for d in data]
    correct = sum(1 for l, p in zip(labels, preds) if l == p)
    total = len(labels)
    acc = correct / total if total > 0 else 0.0

    fig, ax = plt.subplots(figsize=(4,3))
    ax.bar(["Correct", "Wrong"], [correct, total-correct], color=["green", "red"])
    ax.set_title(f"Точность: {acc*100:.2f}% (из {total} пометок)")
    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return buf

# ===================== ХЭНДЛЕРЫ =====================

@dp.message_handler(commands=["start"])
async def cmd_start(message: types.Message):
    user_id = message.from_user.id
    username = message.from_user.username or message.from_user.full_name or "Пользователь"
    # Запишем или обновим пользователя
    cur.execute("INSERT OR IGNORE INTO users (user_id, username, start_ts) VALUES (?, ?, ?)",
                (user_id, username, datetime.utcnow().isoformat()))
    conn.commit()

    text = (f"👋 Привет, <b>{username}</b>!\n\n"
            "Я бот сигналов по валютным парам.\n"
            "Автор: Nurik\n\n"
            "Команды:\n"
            "/новый - получить новый торговый сигнал\n"
            "/статистика - посмотреть статистику точности\n"
            "/пара - выбрать валютную пару\n"
            "/таймфрейм - выбрать таймфрейм\n\n"
            "Используй кнопки ниже для управления.")
    await message.answer(text, reply_markup=settings_keyboard(user_id))

@dp.message_handler(commands=["новый"])
async def cmd_new(message: types.Message):
    user_id = message.from_user.id
    pair, timeframe = get_user_settings(user_id)
    await message.answer("Генерирую сигнал... ⏳")
    try:
        closes = await get_closes(pair, timeframe)
    except Exception as e:
        await message.answer(f"Ошибка получения данных: {e}")
        return
    feats = compute_features_from_closes(closes)
    model = load_model(pair, timeframe)
    pred = decide_from_features(feats, model)
    dir_text = "BUY" if pred == 1 else "SELL"
    await send_signal(user_id, dir_text, feats, pair, timeframe)

@dp.message_handler(commands=["статистика"])
async def cmd_stats(message: types.Message):
    user_id = message.from_user.id
    cur.execute("SELECT COUNT(*) FROM signals WHERE chat_id = ?", (user_id,))
    total = cur.fetchone()[0] or 0
    cur.execute("SELECT COUNT(*) FROM signals WHERE chat_id = ? AND label IS NOT NULL", (user_id,))
    labeled = cur.fetchone()[0] or 0
    cur.execute("SELECT COUNT(*) FROM signals WHERE chat_id = ? AND label = predicted AND label IS NOT NULL", (user_id,))
    correct = cur.fetchone()[0] or 0
    acc = (correct / labeled * 100) if labeled > 0 else 0.0

    text = (f"📊 <b>Статистика по вашим сигналам</b>\n\n"
            f"Всего сигналов: {total}\n"
            f"Отмечено: {labeled}\n"
            f"Корректных по пометкам: {correct}\n"
            f"Точность: {acc:.2f}%")

    chart = plot_accuracy_chart()
    if chart:
        await bot.send_photo(user_id, chart, caption=text, parse_mode="HTML")
    else:
        await message.answer(text, parse_mode="HTML")

@dp.message_handler(commands=["пара"])
async def cmd_pair(message: types.Message):
    user_id = message.from_user.id
    kb = InlineKeyboardMarkup(row_width=1)
    for p in PAIRS:
        kb.insert(InlineKeyboardButton(p, callback_data=f"setpair|{p}|{user_id}"))
    await message.answer("Выберите валютную пару:", reply_markup=kb)

@dp.message_handler(commands=["таймфрейм"])
async def cmd_timeframe(message: types.Message):
    user_id = message.from_user.id
    kb = InlineKeyboardMarkup(row_width=1)
    for tf in TIMEFRAMES:
        kb.insert(InlineKeyboardButton(tf, callback_data=f"settf|{tf}|{user_id}"))
    await message.answer("Выберите таймфрейм:", reply_markup=kb)

# Обработка кнопок
@dp.callback_query_handler(lambda c: True)
async def callbacks_handler(callback: types.CallbackQuery):
    data = callback.data
    user_id = callback.from_user.id

    if data.startswith("fb|"):
        label = int(data.split("|")[1])
        msg_id = callback.message.message_id
        cur.execute("SELECT id, features, pair, timeframe FROM signals WHERE message_id = ? AND chat_id = ? ORDER BY id DESC LIMIT 1",
                    (msg_id, user_id))
        row = cur.fetchone()
        if not row:
            await callback.answer("Сигнал не найден в БД", show_alert=True)
            return
        sig_id, feats_json, pair, timeframe = row
        feats = np.array(json.loads(feats_json))
        # Обновляем метку
        cur.execute("UPDATE signals SET label = ? WHERE id = ?", (label, sig_id))
        conn.commit()

        # Обучаем модель онлайн
        model = load_model(pair, timeframe)
        try:
            model.partial_fit(feats, [label])
            save_model(model, pair, timeframe)
        except Exception:
            pass
        await callback.answer("Фидбек сохранён ✅")

    elif data.startswith("setpair|"):
        _, pair, uid_str = data.split("|")
        if int(uid_str) != user_id:
            await callback.answer("Это не для вас!", show_alert=True)
            return
        set_user_pair(user_id, pair)
        await callback.answer(f"Пара изменена на {pair}")
        await bot.send_message(user_id, f"Ваша новая валютная пара: {pair}")

    elif data.startswith("settf|"):
        _, tf, uid_str = data.split("|")
        if int(uid_str) != user_id:
            await callback.answer("Это не для вас!", show_alert=True)
            return
        set_user_timeframe(user_id, tf)
        await callback.answer(f"Таймфрейм изменён на {tf}")
        await bot.send_message(user_id, f"Ваш новый таймфрейм: {tf}")

    elif data.startswith("newsignal|"):
        _, uid_str = data.split("|")
        if int(uid_str) != user_id:
            await callback.answer("Это не для вас!", show_alert=True)
            return
        pair, timeframe = get_user_settings(user_id)
        await callback.answer("Генерирую сигнал... ⏳", show_alert=False)
        try:
            closes = await get_closes(pair, timeframe)
        except Exception as e:
            await bot.send_message(user_id, f"Ошибка получения данных: {e}")
            return
        feats = compute_features_from_closes(closes)
        model = load_model(pair, timeframe)
        pred = decide_from_features(feats, model)
        dir_text = "BUY" if pred == 1 else "SELL"
        await send_signal(user_id, dir_text, feats, pair, timeframe)

    elif data.startswith("stats|"):
        _, uid_str = data.split("|")
        if int(uid_str) != user_id:
            await callback.answer("Это не для вас!", show_alert=True)
            return
        # Показать статистику как в команде /статистика
        cur.execute("SELECT COUNT(*) FROM signals WHERE chat_id = ?", (user_id,))
        total = cur.fetchone()[0] or 0
        cur.execute("SELECT COUNT(*) FROM signals WHERE chat_id = ? AND label IS NOT NULL", (user_id,))
        labeled = cur.fetchone()[0] or 0
        cur.execute("SELECT COUNT(*) FROM signals WHERE chat_id = ? AND label = predicted AND label IS NOT NULL", (user_id,))
        correct = cur.fetchone()[0] or 0
        acc = (correct / labeled * 100) if labeled > 0 else 0.0

        text = (f"📊 <b>Статистика по вашим сигналам</b>\n\n"
                f"Всего сигналов: {total}\n"
                f"Отмечено: {labeled}\n"
                f"Корректных по пометкам: {correct}\n"
                f"Точность: {acc:.2f}%")
        chart = plot_accuracy_chart()
        if chart:
            await bot.send_photo(user_id, chart, caption=text, parse_mode="HTML")
        else:
            await bot.send_message(user_id, text, parse_mode="HTML")
        await callback.answer()

# ===================== ЗАПУСК БОТА =====================
if __name__ == "__main__":
    print("Bot started...")
    executor.start_polling(dp, skip_updates=True)
