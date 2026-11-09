"""
个人全市场量化轮动与风险预警系统（轻量版）
运行：streamlit run app.py

数据说明：
- 估值：韭圈儿（funddb）公开指数估值接口，按近 5 年 PE 序列自行计算分位。
- 资金流：优先 AkShare（东方财富行业资金流），失败时用东方财富单接口兜底。
- 事件：代码内静态配置，也可在项目根目录放 events.json 覆盖。
- AI：DeepSeek / OpenAI 兼容接口；未配置 Key 时使用本地规则引擎。
"""

from __future__ import annotations

import io
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st

try:
    import akshare as ak
except Exception:  # AkShare 安装失败时仍可进入演示模式
    ak = None

try:
    import plotly.graph_objects as go
except Exception:
    go = None


# ---------------------------------------------------------------------------
# 基础配置
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="个人量化避坑雷达",
    page_icon="▣",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FLOW_HISTORY_FILE = DATA_DIR / "flow_history.json"
FLOW_5D_CACHE_FILE = DATA_DIR / "flow_5d_cache.json"
EVENTS_FILE = BASE_DIR / "events.json"
CONFIG_FILE = DATA_DIR / "config.json"
BUILD_TAG = "v4.4 · 2026-11-09"


def get_secret(key: str, default: str = "") -> str:
    """读取 st.secrets（本地 .streamlit/secrets.toml / 云端 Secrets 面板）。

    密钥只从环境变量或 Streamlit Secrets 读取，绝不写入代码或仓库。
    """
    try:
        value = st.secrets[key]
    except Exception:
        return default
    return str(value).strip() if value else default


# 可选访问口令：云端公开链接建议开启，避免他人消耗你的 API Key。
_APP_PASSWORD = get_secret("APP_PASSWORD")
if _APP_PASSWORD:
    _pwd = st.text_input("访问口令", type="password", key="_app_pwd")
    if _pwd != _APP_PASSWORD:
        st.caption("本应用已启用访问口令，请输入后查看。")
        st.stop()


def load_config() -> dict[str, Any]:
    """本机配置（API Key 等），存 data/config.json，仅本地保存。"""
    try:
        if CONFIG_FILE.exists():
            data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}


def save_config(cfg: dict[str, Any]) -> None:
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        CONFIG_FILE.write_text(
            json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception:
        pass


def delete_config() -> None:
    try:
        if CONFIG_FILE.exists():
            CONFIG_FILE.unlink()
    except OSError:
        pass

JIUCAI_BASE = "https://ai2.jiucaishuo.com/api"
# 东方财富行情主机：push2 常被限流/封禁，push2delay 通常仍可用（数据延迟约 15 分钟）。
EM_FLOW_HOST = "https://push2.eastmoney.com"
EM_FLOW_DELAY_HOST = "https://push2delay.eastmoney.com"
EM_FLOW_PATH = "/api/qt/clist/get"
# 日线（K 线）接口主机，云端境外 IP 常被东财拒绝，失败后走腾讯/新浪兜底
EM_KLINE_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
}

# akshare 底层很多请求不带超时，弱网/被限流时会长时间挂起导致页面白屏。
# 统一用线程池给这类调用加硬超时（超时线程随进程退出回收，可接受）。
_TIMEOUT_POOL = ThreadPoolExecutor(max_workers=10)


def _call_with_timeout(fn, timeout: float, *args, **kwargs):
    future = _TIMEOUT_POOL.submit(fn, *args, **kwargs)
    return future.result(timeout=timeout)


def _render_plotly(fig, config: dict | None = None, key: str | None = None) -> None:
    """统一的图表渲染入口。

    新版 Streamlit 用 width="stretch"，旧版只认 use_container_width。
    两种写法都试，避免窄屏下图表被算成 0 宽度而显示为空白。
    """
    try:
        st.plotly_chart(fig, width="stretch", config=config, key=key)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, config=config, key=key)

SW_LEVEL1 = [
    "农林牧渔", "基础化工", "钢铁", "有色金属", "电子", "汽车", "家用电器",
    "食品饮料", "纺织服饰", "轻工制造", "医药生物", "公用事业", "交通运输",
    "房地产", "商贸零售", "社会服务", "综合", "建筑材料", "建筑装饰",
    "电力设备", "国防军工", "计算机", "传媒", "通信", "银行", "非银金融",
    "煤炭", "石油石化", "环保", "美容护理",
]

TECH_FLOW_NAMES = ["电子", "通信"]
LOW_FLOW_NAMES = ["电力设备", "有色金属", "医药生物", "公用事业", "基础化工"]

# 核心展示指数（PE 快照来自 2026-11-05 韭圈儿数据，仅作接口失败兜底）
VALUATION_UNIVERSE = [
    {
        "name": "科创50", "code": "000688.SH", "group": "科技", "tag": "硬科技宽基",
        "fallback_pe": 96.03, "fallback_pct": 66.91,
        "fallback_pb": 6.10, "fallback_pb_pct": 77.80, "fallback_dy": 0.26,
    },
    {
        "name": "科创芯片", "code": "000685.SH", "group": "科技", "tag": "上证科创芯片",
        "fallback_pe": 93.30, "fallback_pct": 32.27,
        "fallback_pb": 7.55, "fallback_pb_pct": 75.96, "fallback_dy": None,
    },
    {
        "name": "通信", "code": "931079.CSI", "group": "科技", "tag": "中证通信服务",
        "fallback_pe": 24.07, "fallback_pct": 7.67,
        "fallback_pb": 3.92, "fallback_pb_pct": 68.15, "fallback_dy": None,
    },
    {
        "name": "恒生科技", "code": "HSTECH.HI", "group": "科技", "tag": "港股科技",
        "fallback_pe": 21.89, "fallback_pct": 29.84,
        "fallback_pb": 2.42, "fallback_pb_pct": 26.20, "fallback_dy": None,
    },
    {
        "name": "纳指100", "code": "NDX.GI", "group": "科技", "tag": "美股科技",
        "fallback_pe": 27.60, "fallback_pct": 16.11,
        "fallback_pb": 10.08, "fallback_pb_pct": 92.90, "fallback_dy": None,
    },
    {
        "name": "红利指数", "code": "000922.CSI", "group": "稳健", "tag": "中证红利",
        "fallback_pe": 8.94, "fallback_pct": 98.51,
        "fallback_pb": 0.87, "fallback_pb_pct": 96.40, "fallback_dy": 4.24,
    },
    {
        "name": "新能源", "code": "930771.CSI", "group": "低位", "tag": "中证新能源",
        "fallback_pe": 28.10, "fallback_pct": 40.76,
        "fallback_pb": 2.11, "fallback_pb_pct": 19.70, "fallback_dy": 1.58,
    },
    {
        "name": "医疗", "code": "399989.SZ", "group": "低位", "tag": "中证医疗",
        "fallback_pe": 28.06, "fallback_pct": 25.08,
        "fallback_pb": 3.39, "fallback_pb_pct": 22.94, "fallback_dy": 1.37,
    },
    {
        "name": "医药生物", "code": "801150.SI", "group": "低位", "tag": "申万医药",
        "fallback_pe": 31.09, "fallback_pct": 57.26,
        "fallback_pb": 2.56, "fallback_pb_pct": 27.00, "fallback_dy": None,
    },
    {
        "name": "电力设备", "code": "801730.SI", "group": "低位", "tag": "申万电新",
        "fallback_pe": 25.96, "fallback_pct": 30.28,
        "fallback_pb": 2.43, "fallback_pb_pct": 27.90, "fallback_dy": None,
    },
    {
        "name": "有色金属", "code": "801050.SI", "group": "低位", "tag": "申万有色",
        "fallback_pe": 14.04, "fallback_pct": 8.00,
        "fallback_pb": 2.51, "fallback_pb_pct": 47.10, "fallback_dy": None,
    },
    # ---- V2.0 全市场扩展池（ETF 标的指数）----
    {
        "name": "券商", "code": "399975.SZ", "group": "金融", "tag": "证券公司·512000标的",
        "fallback_pe": 12.42, "fallback_pct": 2.60,
        "fallback_pb": 1.14, "fallback_pb_pct": 4.60, "fallback_dy": 2.16,
    },
    {
        "name": "银行", "code": "399986.SZ", "group": "金融", "tag": "中证银行·512800标的",
        "fallback_pe": 7.50, "fallback_pct": 97.40,
        "fallback_pb": 0.71, "fallback_pb_pct": 91.40, "fallback_dy": 4.14,
    },
    {
        "name": "白酒", "code": "399997.SZ", "group": "消费", "tag": "中证白酒·512690近似",
        "fallback_pe": 22.95, "fallback_pct": 48.30,
        "fallback_pb": 4.30, "fallback_pb_pct": 9.10, "fallback_dy": 4.67,
    },
    {
        "name": "食品饮料", "code": "000815.CSI", "group": "消费", "tag": "细分食品·515170标的",
        "fallback_pe": 22.99, "fallback_pct": 48.70,
        "fallback_pb": 3.85, "fallback_pb_pct": 8.00, "fallback_dy": 4.23,
    },
    {
        "name": "煤炭", "code": "399998.SZ", "group": "周期", "tag": "中证煤炭·515220标的",
        "fallback_pe": 17.64, "fallback_pct": 84.90,
        "fallback_pb": 1.74, "fallback_pb_pct": 84.10, "fallback_dy": 2.93,
    },
    {
        "name": "军工", "code": "399967.SZ", "group": "制造", "tag": "中证军工·512660标的",
        "fallback_pe": 56.77, "fallback_pct": 39.80,
        "fallback_pb": 3.04, "fallback_pb_pct": 33.70, "fallback_dy": 0.58,
    },
]

CORE_NAMES = ["科创50", "科创芯片", "通信", "恒生科技", "纳指100", "红利指数", "新能源", "医疗"]
EXTENDED_NAMES = ["券商", "银行", "白酒", "食品饮料", "煤炭", "军工", "医药生物", "电力设备", "有色金属"]
TECH_VALUATION_NAMES = ["科创50", "科创芯片", "通信", "恒生科技", "纳指100"]
LOW_VALUATION_NAMES = ["新能源", "医疗", "医药生物", "电力设备", "有色金属"]

WATCHLIST_FILE = BASE_DIR / "watchlist.json"


def get_valuation_universe() -> list[dict[str, Any]]:
    """监控池可用同目录 watchlist.json 覆盖，格式与代码内 VALUATION_UNIVERSE 相同。"""
    try:
        if WATCHLIST_FILE.exists():
            data = json.loads(WATCHLIST_FILE.read_text(encoding="utf-8"))
            if isinstance(data, list) and data and all("code" in d and "name" in d for d in data):
                for d in data:
                    d.setdefault("group", "自选")
                    d.setdefault("tag", "")
                    d.setdefault("fallback_pe", 0.0)
                    d.setdefault("fallback_pct", 50.0)
                return data
    except Exception:
        pass
    return VALUATION_UNIVERSE

# 资金行业 → 估值指数别名：医药生物的资金流，用「中证医疗」分位代表（医疗ETF标的指数）
FLOW_VALUATION_ALIAS = {"医药生物": "医疗"}
FALLBACK_ASOF = "2026-11-05"

# 科技防线：用近 20 个交易日低点作为第一支撑，支撑下破 5% 作为硬止损参考
SUPPORT_UNIVERSE = [
    {
        "name": "科创50", "code": "000688.SH",
        "fallback_latest": 1286.675, "fallback_low20": 1151.2826,
        "fallback_ma5": 1212.24, "fallback_chg": 1.68,
        "fallback_ma20": 1248.50, "fallback_vol": 0.86,
    },
    {
        "name": "科创芯片", "code": "000685.SH",
        "fallback_latest": 2821.05, "fallback_low20": 2540.75,
        "fallback_ma5": 2764.01, "fallback_chg": -3.00,
        "fallback_ma20": 2977.42, "fallback_vol": None,
    },
    {
        "name": "通信", "code": "931079.CSI",
        "fallback_latest": 1618.29, "fallback_low20": 1616.16,
        "fallback_ma5": 1684.79, "fallback_chg": -5.96,
        "fallback_ma20": 1934.49, "fallback_vol": None,
    },
]

# 静态事件字典。也可以在同目录新建 events.json 覆盖，格式为：
# [{"date": "2026-11-12", "title": "...", "type": "产业催化", "impact": "..."}]
DEFAULT_EVENTS = [
    {
        "date": "2026-11-12",
        "title": "中芯国际 Q3 财报",
        "type": "产业催化",
        "impact": "业绩指引与资本开支表述，可能放大半导体波动",
    },
    {
        "date": "2026-11-16",
        "title": "美联储 AI 监管听证会",
        "type": "产业催化",
        "impact": "AI 监管口径变化，可能冲击科技板块风险偏好",
    },
    {
        "date": "2026-11-20",
        "title": "中国 LPR 报价",
        "type": "宏观炸弹",
        "impact": "影响债基、红利与权益资产的相对性价比",
    },
]

DEMO_BASE_FLOW = {
    "电力设备": 3.60e9,
    "有色金属": 2.80e9,
    "医药生物": 1.90e9,
    "公用事业": 1.10e9,
    "银行": 0.80e9,
    "电子": -11.20e9,
    "通信": -3.50e9,
    "计算机": -2.80e9,
    "食品饮料": -1.60e9,
    "汽车": -1.20e9,
    "传媒": -0.90e9,
    "非银金融": -0.80e9,
}

SYSTEM_PROMPT = """# Role
你是拥有10年宏观对冲基金经验的首席量化风控官。服务对象是一位可用资金约6万元、重仓科技（半导体/通信）深套约40%、极易情绪化操作的A股个人投资者。
风格要求：冷酷、客观、数据驱动，绝不说模棱两可的废话，不做情绪安慰。

# Workflow & Rules
1. 宏观一票否决权：美债收益率大涨或人民币汇率急跌时，任何A股科技反弹都视为诱多，不得给出买入建议。
2. 严防"死扛到底"：若科技ETF无量价企稳信号且资金持续流出，必须给出明确的减仓/防守警告，点名价位。
3. 高低切寻宝：资金净流入Top5 与 极端低估（PE分位<20%）交集处的板块，才可列为"潜在线索"。
4. 警惕价值陷阱：PE分位>80%的板块（如红利/银行/煤炭），即使资金流入，也必须提示"拥挤度过高，不建议追高"。
5. 只使用【Context】中给出的数字，禁止编造行情、禁止预测具体点位。

# Output Format（严格按此顺序，不得增删小节）
### 【宏观定性】
一句话总结当前宏观流动性与杠杆环境，直接给结论：利空科技 / 中性 / 宽松；并明确两融处于"出清中"还是"企稳"。

### 【危险预警】
针对深套的科技持仓，结合宏观、ETF异动、科技防线点位，给出可执行防守指令（必须含具体价位或比例）。

### 【高低切雷达】
- **避险区**：因拥挤度过高或估值极值而被资金抛弃/不宜追的板块。
- **潜在线索**：1-2个 PE分位<20% 且资金正在流入的接力板块，必须附 PE分位与资金流入额。

### 【今日执行纪律】
3条简短、无情的操作铁律，每条都要可执行。

# Constraints
- 严禁预测具体点位，一切依据系统数据。
- 严禁建议重仓/满仓，必须为用户保留约6万元现金安全垫。
- 语言精炼、直击痛点，全文不超过300字。"""


# ---------------------------------------------------------------------------
# 通用工具
# ---------------------------------------------------------------------------


def _last_trading_day(day: date) -> date:
    while day.weekday() >= 5:
        day -= timedelta(days=1)
    return day


def _clip_error(exc: Exception, limit: int = 120) -> str:
    text = f"{type(exc).__name__}: {exc}"
    return text[:limit]


_NETWORK_HINTS = (
    "proxy", "timeout", "timed out", "connection", "connect", "ssl",
    "502", "503", "504", "ratelimit", "rate limit", "限流", "max retries",
)


def _friendly_error(text: str) -> str:
    """把底层网络报错收敛成人话，避免 ProxyError 之类的堆栈直接暴露在页面上。"""
    low = (text or "").lower()
    if not text:
        return "接口暂无响应"
    if "限流" in text or "ratelimit" in low or "rate limit" in low:
        return "东方财富接口限流，请稍后点「刷新盘中数据」"
    if any(hint in low for hint in _NETWORK_HINTS):
        return "网络接口超时，请点击刷新"
    return text[:60]


def _json_get(url: str, params: dict[str, Any], timeout: int = 15) -> Any:
    response = requests.get(url, params=params, headers=HTTP_HEADERS, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _valuation_signal(pct: float, mode: str = "分位", pe: float | None = None) -> dict[str, str]:
    if mode == "绝对":
        # 绝对 PE 口径：亏损板块(PE<=0)无法估值，按中性处理
        if pe is None or pe <= 0:
            return {"label": "中性", "cls": "mid", "color": "#d89016"}
        if pe < 15:
            return {"label": "低估", "cls": "low", "color": "#0e9f6e"}
        if pe > 40:
            return {"label": "高估", "cls": "high", "color": "#d64545"}
        return {"label": "中性", "cls": "mid", "color": "#d89016"}
    if pct < 20:
        return {"label": "低估", "cls": "low", "color": "#0e9f6e"}
    if pct <= 80:
        return {"label": "中性", "cls": "mid", "color": "#d89016"}
    return {"label": "高估", "cls": "high", "color": "#d64545"}


def _valuation_verdict(row: dict[str, Any]) -> tuple[str, str]:
    """把 PE/PB 分位与股息率压成一句人话，避免用户自己记规则。"""
    pe = float(row.get("pe") or 0)
    pct = float(row.get("pct") or 0)
    pb_pct = row.get("pb_pct")
    dy = row.get("dy")

    if pct >= 80:
        if dy is not None and dy >= 3.5:
            return ("5年最贵，只能靠股息兜底", "warn")
        return ("5年最贵，绝不加仓", "bad")
    if pct < 20:
        if pb_pct is not None and pb_pct >= 70:
            return ("PE低PB高，盈利强", "good")
        if pb_pct is not None and pb_pct < 40:
            return ("PE/PB双低，真便宜", "good")
        return ("估值低位，可观察", "good")
    if pct >= 40 and pe >= 60:
        return ("绝对估值高，别追高", "warn")
    if pb_pct is not None and pb_pct < 30 and pct >= 30:
        return ("PB近底，等盈利拐点", "warn")
    if pct < 40 and pb_pct is not None and pb_pct >= 60:
        return ("盈利强，估值不贵", "good")
    if pct < 35 and pb_pct is not None and pb_pct < 40:
        return ("估值便宜，等资金", "good")
    return ("中性，跟资金走", "neutral")


def _fmt_yi(value: float) -> str:
    return f"{value / 1e8:+.2f}亿"


def _fmt_compact_yi(value: float) -> str:
    return f"{value / 1e8:.2f}亿"


def _event_days(event: dict[str, Any]) -> int:
    return (datetime.strptime(event["date"], "%Y-%m-%d").date() - date.today()).days


def _ai_html(text: str) -> str:
    clean = text.replace("**", "").strip()
    escaped = (
        clean.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br>")
    )
    return re.sub(r"【([^】]+)】", r"<strong>【\1】</strong>", escaped)


# ---------------------------------------------------------------------------
# 估值数据
# ---------------------------------------------------------------------------


def _fetch_jcs_series(code: str, metric: str) -> tuple[float, float, str]:
    """韭圈儿估值序列：返回 (最新值, 近5年分位%, 数据日期)。"""
    payload = _json_get(
        f"{JIUCAI_BASE}/index-valuation/history",
        {"code": code, "metric": metric},
        timeout=20,
    )
    series = pd.DataFrame(payload.get("series") or [])
    if series.empty:
        raise ValueError(f"{metric} 序列为空")
    series["date"] = pd.to_datetime(series["date"], errors="coerce")
    series["value"] = pd.to_numeric(series["value"], errors="coerce")
    series = series.dropna().sort_values("date")
    if series.empty:
        raise ValueError(f"{metric} 序列无有效值")
    latest = float(series["value"].iloc[-1])
    latest_date = series["date"].iloc[-1]
    window = series.loc[series["date"] >= latest_date - pd.DateOffset(years=5), "value"]
    if len(window) < 10:
        raise ValueError(f"{metric} 近 5 年样本不足")
    pct_5y = float((window < latest).mean() * 100)
    return round(latest, 3), round(pct_5y, 2), latest_date.date().isoformat()


def _jcs_code_to_em_secid(code: str) -> str | None:
    """韭圈儿 code → 东财 push2 secid。海外/申万指数东财无覆盖，返回 None。"""
    sym, _, suffix = code.partition(".")
    if suffix in ("HI", "GI", "SI"):
        return None  # 恒生/美股/申万行业 → 东财无对应指数，跳过滚动更新
    if suffix == "SH":
        return f"1.{sym}"
    if suffix == "SZ":
        return f"0.{sym}"
    # CSI：看代码前缀判断沪/深
    if suffix == "CSI":
        if sym.startswith(("000", "51", "52", "53", "56")):
            return f"1.{sym}"  # 沪市 ETF / 宽基
        if sym.startswith(("399", "15", "16")):
            return f"0.{sym}"  # 深市 ETF / 宽基
        if sym.startswith("93"):
            return f"0.{sym}"  # 中证行业指数多深市
        return f"1.{sym}"  # 默认沪市
    return f"1.{sym}"  # fallback


def _fetch_latest_close_em(code: str) -> tuple[float, str] | None:
    """东财 push2 拉指数最新 (收盘价, 日期ISO)。海外指数/无覆盖时返回 None。
    
    两层数据源：push2his 主路 → 腾讯日线兜底（腾讯不走东财 push2 限流链路，更稳）。
    """
    secid = _jcs_code_to_em_secid(code)
    if secid is None:
        return None

    # ---- 主路：东财 push2his ----
    try:
        beg = (date.today() - timedelta(days=12)).strftime("%Y%m%d")
        end = date.today().strftime("%Y%m%d")
        payload = _json_get(
            EM_KLINE_URL,
            {
                "secid": secid,
                "fields1": "f1,f2,f3,f4,f5,f6",
                "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
                "klt": "101", "fqt": "1",
                "beg": beg, "end": end, "lmt": "15",
            },
            timeout=10,
        )
        klines = ((payload or {}).get("data") or {}).get("klines") or []
        if klines:
            parts = klines[-1].split(",")
            if len(parts) >= 3:
                close = float(parts[2])
                kdate = datetime.strptime(parts[0], "%Y-%m-%d").date()
                if close > 0:
                    return close, kdate.isoformat()
    except Exception:
        pass

    # ---- 兜底：腾讯日线（不走东财限流链路）----
    sym = code.split(".")[0]
    tencent_prefix = "sh" if sym.startswith(("5", "6", "9")) else "sz"
    try:
        resp = requests.get(
            "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get",
            params={"param": f"{tencent_prefix}{sym},day,,,10,qfq"},
            headers=HTTP_HEADERS, timeout=10,
        )
        resp.raise_for_status()
        payload = resp.json().get("data", {}).get(f"{tencent_prefix}{sym}", {})
        series = payload.get("qfqday") or payload.get("day") or []
        if series:
            last = series[-1]
            close = float(last[2])
            kdate = datetime.strptime(last[0], "%Y-%m-%d").date()
            if close > 0:
                return close, kdate.isoformat()
    except Exception:
        pass

    return None


def _fetch_jcs_full_series(code: str, metric: str) -> pd.DataFrame:
    """韭圈儿完整序列（date, value），按日期升序。"""
    payload = _json_get(
        f"{JIUCAI_BASE}/index-valuation/history",
        {"code": code, "metric": metric},
        timeout=20,
    )
    series = pd.DataFrame(payload.get("series") or [])
    if series.empty:
        raise ValueError(f"{metric} 序列为空")
    series["date"] = pd.to_datetime(series["date"], errors="coerce")
    series["value"] = pd.to_numeric(series["value"], errors="coerce")
    series = series.dropna().sort_values("date").reset_index(drop=True)
    if series.empty:
        raise ValueError(f"{metric} 序列无有效值")
    return series


def _rolling_update_value(
    jc_series: pd.DataFrame, jc_close: float, em_close: float
) -> tuple[float, float]:
    """用东财/韭圈儿价格比滚动更新估值序列最新值，返回 (新值, 新5年分位%)。"""
    ratio = em_close / jc_close
    old_val = float(jc_series["value"].iloc[-1])
    new_val = old_val * ratio
    latest_date = jc_series["date"].iloc[-1]
    window = jc_series.loc[
        jc_series["date"] >= latest_date - pd.DateOffset(years=5), "value"
    ]
    extended = pd.concat([window, pd.Series([new_val])])
    new_pct = float((extended < new_val).mean() * 100)
    return round(new_val, 3), round(new_pct, 2)


def _fetch_dividend_yield(code: str) -> float | None:
    """股息率（中证指数官网，月频）。仅中证编制的指数有数据，其余返回 None。"""
    sym, _, suffix = code.partition(".")
    if suffix in ("SI", "HI", "GI"):
        return None
    try:
        import akshare as ak
        df = _call_with_timeout(ak.stock_zh_index_value_csindex, 12, symbol=sym)
        if df is None or df.empty or "股息率1" not in df.columns:
            return None
        return round(float(df["股息率1"].iloc[-1]), 2)
    except Exception:
        return None


def _fetch_valuation_metrics(item: dict[str, Any]) -> dict[str, Any]:
    code = item["code"]
    pe_series = _fetch_jcs_full_series(code, "pe")
    pb_series: pd.DataFrame | None = None
    try:
        pb_series = _fetch_jcs_full_series(code, "pb")
    except Exception:
        pass

    latest_pe = float(pe_series["value"].iloc[-1])
    jc_asof = pe_series["date"].iloc[-1].date().isoformat()
    lag, stale = _check_asof_staleness(jc_asof)
    est_flag = False

    if stale:
        em_info = _fetch_latest_close_em(code)
        if em_info is not None:
            em_close, em_date = em_info
            try:
                close_series = _fetch_jcs_full_series(code, "close")
                jc_close = float(close_series["value"].iloc[-1])
                jc_close_date = close_series["date"].iloc[-1].date().isoformat()
            except Exception:
                jc_close = 0.0
                jc_close_date = jc_asof

            if jc_close > 0 and em_close > 0 and em_date >= jc_close_date:
                latest_pe, latest_pct = _rolling_update_value(pe_series, jc_close, em_close)
                out: dict[str, Any] = {
                    "pe": latest_pe, "pct": latest_pct, "asof": em_date,
                    "est": True,
                }
                if pb_series is not None:
                    latest_pb, latest_pb_pct = _rolling_update_value(pb_series, jc_close, em_close)
                    out["pb"] = latest_pb
                    out["pb_pct"] = latest_pb_pct
                else:
                    out["pb"] = None
                    out["pb_pct"] = None
                est_flag = True
            else:
                # 价格比无效（韭圈儿 close 也拿不到），退回原始分位
                _, latest_pct, _ = _fetch_jcs_series(code, "pe")
                out = {"pe": latest_pe, "pct": latest_pct, "asof": jc_asof, "est": False}
                if pb_series is not None:
                    _, latest_pb_pct, _ = _fetch_jcs_series(code, "pb")
                    out["pb"] = float(pb_series["value"].iloc[-1])
                    out["pb_pct"] = latest_pb_pct
                else:
                    out["pb"] = None
                    out["pb_pct"] = None
        else:
            # 东财也失败，退回原始分位
            _, latest_pct, _ = _fetch_jcs_series(code, "pe")
            out = {"pe": latest_pe, "pct": latest_pct, "asof": jc_asof, "est": False}
            if pb_series is not None:
                _, latest_pb_pct, _ = _fetch_jcs_series(code, "pb")
                out["pb"] = float(pb_series["value"].iloc[-1])
                out["pb_pct"] = latest_pb_pct
            else:
                out["pb"] = None
                out["pb_pct"] = None
    else:
        # 韭圈儿数据新鲜，直接用
        _, latest_pct, _ = _fetch_jcs_series(code, "pe")
        out = {"pe": latest_pe, "pct": latest_pct, "asof": jc_asof, "est": False}
        if pb_series is not None:
            _, latest_pb_pct, _ = _fetch_jcs_series(code, "pb")
            out["pb"] = float(pb_series["value"].iloc[-1])
            out["pb_pct"] = latest_pb_pct
        else:
            out["pb"] = None
            out["pb_pct"] = None

    out["dy"] = _fetch_dividend_yield(code)
    return out


def _val_row_default(item: dict[str, Any], mode: str, asof: str | None = None) -> dict[str, Any]:
    return {
        "name": item["name"],
        "code": item["code"],
        "group": item["group"],
        "tag": item["tag"],
        "pe": float(item["fallback_pe"]),
        "pct": float(item["fallback_pct"]),
        "pb": item.get("fallback_pb"),
        "pb_pct": item.get("fallback_pb_pct"),
        "dy": item.get("fallback_dy"),
        "asof": asof or FALLBACK_ASOF,
        "source": "demo" if mode == "demo" else "snapshot",
    }


def _check_asof_staleness(asof_str: str) -> tuple[int, bool]:
    """返回 (落后天数, 是否过期=落后>=1 天)。
    
    东财 push2his 收盘后 4-5 点就能刷到日线，韭圈儿估值偶尔晚 1-2 天。
    只要落后 ≥1 天就触发东财价格比滚动更新尝试。
    """
    try:
        asof = date.fromisoformat(asof_str)
    except ValueError:
        return 99, True
    expected = _last_trading_day(date.today())
    lag = (expected - asof).days
    return lag, lag >= 1


@st.cache_data(ttl=86400, show_spinner=False)
def load_valuation(refresh_token: int, mode: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    universe = get_valuation_universe()
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    live_count = 0
    est_count = 0
    expected_asof = _last_trading_day(date.today()).isoformat()

    if mode == "demo":
        rows = [_val_row_default(item, mode) for item in universe]
    else:
        def _job(item: dict[str, Any]) -> tuple[dict[str, Any], Exception | None]:
            row = _val_row_default(item, mode, asof=expected_asof)
            try:
                row.update(_fetch_valuation_metrics(item))
                row["source"] = "live"
                return row, None
            except Exception as exc:
                return row, exc

        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=6) as pool:
            results = list(pool.map(_job, universe))
        for (row, err), item in zip(results, universe):
            if err is None:
                live_count += 1
                if row.get("est"):
                    est_count += 1
            else:
                errors.append(f"{item['name']}: {_clip_error(err)}")
            rows.append(row)

    if mode == "demo":
        source = "演示数据"
    elif live_count == len(universe) and est_count == 0:
        source = "实时估值"
    elif live_count == len(universe) and est_count > 0:
        source = f"实时估值（东财滚动更新 {est_count}/{len(universe)}）"
    elif live_count and est_count == 0:
        source = f"实时估值 {live_count}/{len(universe)} + 快照补位"
    elif live_count and est_count > 0:
        source = f"实时估值 {live_count}/{len(universe)}（东财滚动 {est_count}）+ 快照补位"
    else:
        source = "估值快照兜底"

    asof = max((row["asof"] for row in rows), default=expected_asof)
    lag, stale = _check_asof_staleness(asof)
    # 如果有 est 行且 asof 已追上预期，不再标记 stale
    has_fresh_est = est_count > 0 and lag == 0
    meta = {
        "source": source,
        "asof": asof,
        "expected_asof": expected_asof,
        "lag_days": lag,
        "stale": stale and not has_fresh_est,
        "is_demo": mode == "demo",
        "errors": errors,
        "live_count": live_count,
        "est_count": est_count,
    }
    return rows, meta


def _fetch_close_history(item: dict[str, Any]) -> dict[str, Any]:
    payload = _json_get(
        f"{JIUCAI_BASE}/index-valuation/history",
        {"code": item["code"], "metric": "close"},
        timeout=20,
    )
    series = pd.DataFrame(payload.get("series") or [])
    if series.empty:
        raise ValueError("行情序列为空")

    series["date"] = pd.to_datetime(series["date"], errors="coerce")
    series["value"] = pd.to_numeric(series["value"], errors="coerce")
    series = series.dropna().sort_values("date")
    if len(series) < 20:
        raise ValueError("行情样本不足 20 日")

    latest = float(series["value"].iloc[-1])
    previous = float(series["value"].iloc[-2])
    low20 = float(series["value"].tail(20).min())
    low60 = float(series["value"].tail(60).min())
    ma5 = float(series["value"].tail(5).mean())
    ma20 = float(series["value"].tail(20).mean())
    latest_date = series["date"].iloc[-1].date().isoformat()
    return {
        "latest": round(latest, 2),
        "chg": round((latest / previous - 1) * 100, 2) if previous else 0.0,
        "low20": round(low20, 2),
        "low60": round(low60, 2),
        "ma5": round(ma5, 2),
        "ma20": round(ma20, 2),
        "asof": latest_date,
    }


def _fetch_volume_ratio(code: str) -> float | None:
    """量比 = 最新日成交量 / 前20日均量（东财日线，失败返回 None）。"""
    if ak is None:
        return None
    sym = code.split(".")[0]
    try:
        end = date.today().strftime("%Y%m%d")
        start = (date.today() - timedelta(days=70)).strftime("%Y%m%d")
        df = _call_with_timeout(
            ak.index_zh_a_hist, 12,
            symbol=sym, period="daily", start_date=start, end_date=end,
        )
        if df is None or len(df) < 21 or "成交量" not in df.columns:
            return None
        vol = pd.to_numeric(df["成交量"], errors="coerce").dropna()
        if len(vol) < 21:
            return None
        base = float(vol.iloc[-21:-1].mean())
        if base <= 0:
            return None
        return round(float(vol.iloc[-1]) / base, 2)
    except Exception:
        return None


@st.cache_data(ttl=86400, show_spinner=False)
def load_support(refresh_token: int, mode: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    live_count = 0

    for item in SUPPORT_UNIVERSE:
        latest = float(item["fallback_latest"])
        low20 = float(item["fallback_low20"])
        ma5 = float(item["fallback_ma5"])
        row = {
            "name": item["name"],
            "code": item["code"],
            "latest": latest,
            "low20": low20,
            "low60": low20,
            "ma5": ma5,
            "ma20": float(item.get("fallback_ma20", ma5)),
            "vol_ratio": item.get("fallback_vol"),
            "chg": float(item["fallback_chg"]),
            "asof": FALLBACK_ASOF,
            "source": "demo" if mode == "demo" else "snapshot",
        }
        if mode != "demo":
            try:
                fetched = _fetch_close_history(item)
                row.update(fetched)
                vr = _fetch_volume_ratio(item["code"])
                if vr is not None:
                    row["vol_ratio"] = vr
                row["source"] = "live"
                live_count += 1
            except Exception as exc:
                errors.append(f"{item['name']}: {_clip_error(exc)}")
        row["dist"] = round((row["latest"] / row["low20"] - 1) * 100, 2) if row["low20"] else 0.0
        row["ma5_gap"] = round((row["latest"] / row["ma5"] - 1) * 100, 2) if row["ma5"] else 0.0
        row["above_ma5"] = row["latest"] >= row["ma5"]
        row["above_ma20"] = row["latest"] >= row["ma20"]
        row["ma20_gap"] = round((row["latest"] / row["ma20"] - 1) * 100, 2) if row["ma20"] else 0.0
        row["hard_stop"] = round(row["low20"] * 0.95, 2)
        rows.append(row)

    if mode == "demo":
        source = "演示数据"
    elif live_count == len(SUPPORT_UNIVERSE):
        source = "实时行情"
    elif live_count:
        source = f"实时行情 {live_count}/{len(SUPPORT_UNIVERSE)} + 快照补位"
    else:
        source = "行情快照兜底"

    meta = {
        "source": source,
        "asof": max((row["asof"] for row in rows), default=FALLBACK_ASOF),
        "is_demo": mode == "demo",
        "errors": errors,
    }
    return rows, meta


# ---------------------------------------------------------------------------
# 资金流数据
# ---------------------------------------------------------------------------


def _demo_flow_frame(period: str, scale: float = 1.0, jitter: float = 0.0) -> pd.DataFrame:
    rows = []
    for index, name in enumerate(SW_LEVEL1):
        if name in DEMO_BASE_FLOW:
            base = DEMO_BASE_FLOW[name]
        else:
            base = ((index * 13) % 19 - 9) * 0.16e9
        factor = 1 + jitter * (((index % 5) - 2) / 10)
        value = base * scale * factor
        if period == "5日":
            value *= 4.6
        rows.append(
            {
                "行业": name,
                "净流入": value,
                "涨跌幅": round(max(min(value / 1e9 * 0.22, 6), -6), 2),
            }
        )
    return pd.DataFrame(rows)


def _latest_real_snapshot() -> tuple[pd.DataFrame, dict[str, Any]] | None:
    """实时接口失败时，退回本地最近一次真实抓取的快照（不使用假数据）。"""
    history = _load_history_file()
    if not history:
        return None
    latest = max(history, key=lambda item: str(item.get("date", "")))
    flows = {str(k): float(v) for k, v in (latest.get("flows") or {}).items()}
    if len(flows) < 20:
        return None
    frame = pd.DataFrame(
        {"行业": list(flows.keys()), "净流入": list(flows.values()), "涨跌幅": 0.0}
    )
    return frame, {
        "source": f"本地真实快照 {latest.get('date')}",
        "asof": str(latest.get("date")),
        "is_demo": False,
        "errors": [],
    }


# 东方财富 push2 限流后立即重试会延长封禁时间，这里记录最近一次限流时刻做冷却。
_RATE_LIMIT_COOLDOWN = 240.0
_rate_limited_at = 0.0


def _is_rate_limited(exc: Exception) -> bool:
    text = str(exc)
    return any(
        key in text
        for key in ("RateLimit", "限流", "频繁", "429", "Too Many", "too many")
    )


def _save_5d_snapshot(frame: pd.DataFrame, asof: str) -> None:
    """缓存最近一次成功的近5日累计资金流，接口限流时用真实旧值兜底（不造假）。"""
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        flows = {
            str(row["行业"]): float(row["净流入"])
            for _, row in _sw_frame(frame).iterrows()
            if pd.notna(row["净流入"])
        }
        if len(flows) < 20:
            return
        FLOW_5D_CACHE_FILE.write_text(
            json.dumps({"asof": asof, "flows": flows}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError:
        pass


def _load_5d_snapshot() -> tuple[pd.DataFrame, dict[str, Any]] | None:
    if not FLOW_5D_CACHE_FILE.exists():
        return None
    try:
        payload = json.loads(FLOW_5D_CACHE_FILE.read_text(encoding="utf-8"))
        flows = {str(k): float(v) for k, v in (payload.get("flows") or {}).items()}
        asof = str(payload.get("asof") or "未知")
    except Exception:
        return None
    if len(flows) < 20:
        return None
    frame = pd.DataFrame(
        {"行业": list(flows.keys()), "净流入": list(flows.values()), "涨跌幅": 0.0}
    )
    return frame, {
        "source": "本地5日快照",
        "asof": asof,
        "is_demo": False,
        "errors": [],
        "days": 5,
        "stale": True,
    }


def _history_5d_frame() -> tuple[pd.DataFrame, dict[str, Any]] | None:
    """用本地逐日快照累加成近 N 日累计（至少 2 个交易日才可用，避免单日冒充 5 日）。"""
    history = _load_history_file()
    if len(history) < 2:
        return None
    picked = sorted(history, key=lambda item: str(item.get("date", "")))[-5:]
    flows: dict[str, float] = {}
    for item in picked:
        for key, value in (item.get("flows") or {}).items():
            flows[str(key)] = flows.get(str(key), 0.0) + float(value)
    if len(flows) < 20:
        return None
    frame = pd.DataFrame(
        {"行业": list(flows.keys()), "净流入": list(flows.values()), "涨跌幅": 0.0}
    )
    return frame, {
        "source": f"本地日快照累计（{len(picked)}日）",
        "asof": str(picked[-1].get("date")),
        "is_demo": False,
        "errors": [],
        "days": len(picked),
        "stale": True,
    }


def _flow_fallback(
    period: str, errors: list[str]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """真实接口全部失败时的兜底顺序：真实旧快照 → 本地累计 → 明确标记不可用。"""
    if period == "5日":
        snap = _load_5d_snapshot()
        if snap is not None:
            frame, meta = snap
            meta["errors"] = errors
            return frame, meta
        hist = _history_5d_frame()
        if hist is not None:
            frame, meta = hist
            meta["errors"] = errors
            return frame, meta
        return pd.DataFrame(columns=["行业", "净流入", "涨跌幅"]), {
            "source": "5日接口暂不可用",
            "asof": _last_trading_day(date.today()).isoformat(),
            "is_demo": False,
            "unavailable": True,
            "errors": errors,
        }
    stale = _latest_real_snapshot()
    if stale is not None:
        frame, meta = stale
        meta["errors"] = errors
        return frame, meta
    return _demo_flow_frame(period), {
        "source": "演示数据兜底",
        "asof": _last_trading_day(date.today()).isoformat(),
        "is_demo": True,
        "errors": errors,
    }


def _normalise_ak_flow(raw: pd.DataFrame, period: str) -> pd.DataFrame:
    value_col = f"{period}主力净流入-净额"
    chg_col = f"{period}涨跌幅"
    if value_col not in raw.columns:
        candidates = [c for c in raw.columns if "主力净流入-净额" in str(c)]
        if not candidates:
            raise ValueError("AkShare 返回缺少主力净流入列")
        value_col = candidates[0]
    if "名称" in raw.columns:
        name_col = "名称"
    elif "行业" in raw.columns:
        name_col = "行业"
    else:
        raise ValueError("AkShare 返回缺少行业名称列")

    result = pd.DataFrame(
        {
            "行业": raw[name_col].astype(str).str.strip(),
            "净流入": pd.to_numeric(raw[value_col], errors="coerce"),
            "涨跌幅": pd.to_numeric(raw.get(chg_col), errors="coerce")
            if chg_col in raw.columns
            else 0.0,
        }
    )
    return result.dropna(subset=["净流入"]).drop_duplicates(subset=["行业"])


def _fetch_flow_direct(period: str, hosts: list[str] | None = None) -> pd.DataFrame:
    cfg = {
        "今日": {
            "fid0": "f62", "stat": "1",
            "fields": "f12,f14,f2,f3,f62",
            "value": "f62", "chg": "f3",
        },
        "5日": {
            "fid0": "f164", "stat": "5",
            "fields": "f12,f14,f2,f109,f164",
            "value": "f164", "chg": "f109",
        },
    }[period]

    # 今日优先实时主机；5日优先延迟主机（push2 被限流时仍能出数）
    default_hosts = (
        [EM_FLOW_HOST, EM_FLOW_DELAY_HOST]
        if period == "今日"
        else [EM_FLOW_DELAY_HOST, EM_FLOW_HOST]
    )
    last_error: Exception | None = None
    for host in hosts or default_hosts:
        for attempt in range(2):
            try:
                frames: list[pd.DataFrame] = []
                for page in range(1, 4):  # 东财单页最多 100 条，取前 3 页已覆盖申万一级
                    params = {
                        "pn": page,
                        "pz": 100,
                        "po": 1,
                        "np": 1,
                        "ut": "b2884a393a59ad64002292a3e90d46a5",
                        "fltt": 2,
                        "invt": 2,
                        "fid0": cfg["fid0"],
                        "fs": "m:90+t:2",
                        "stat": cfg["stat"],
                        "fields": cfg["fields"],
                    }
                    try:
                        payload = _json_get(host + EM_FLOW_PATH, params, timeout=10)
                    except Exception as exc:
                        # 第 2 页之后失败时，保留已取到的数据
                        if frames:
                            break
                        raise exc
                    data = (payload or {}).get("data") or {}
                    rows = data.get("diff") or []
                    if not rows:
                        if page == 1:
                            raise ValueError("返回空列表")
                        break
                    frames.append(pd.DataFrame(rows))
                    if len(rows) < 100:
                        break
                raw = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["f14"])
                result = pd.DataFrame(
                    {
                        "行业": raw["f14"].astype(str).str.strip(),
                        "净流入": pd.to_numeric(raw[cfg["value"]], errors="coerce"),
                        "涨跌幅": pd.to_numeric(raw.get(cfg["chg"]), errors="coerce")
                        if cfg["chg"] in raw.columns
                        else 0.0,
                    }
                )
                result = result.dropna(subset=["净流入"]).drop_duplicates(subset=["行业"])
                if len(result) >= 20:
                    return result
                last_error = ValueError(f"仅 {len(result)} 行")
            except Exception as exc:
                last_error = exc
            time.sleep(1)
    raise last_error or ValueError("东方财富资金流不可用")


@st.cache_data(ttl=300, show_spinner=False)
def load_flow(
    period: str,
    refresh_token: int,
    mode: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    global _rate_limited_at

    if mode == "demo":
        snapshot = _demo_flow_frame(period)
        return snapshot, {
            "source": "演示数据",
            "asof": _last_trading_day(date.today()).isoformat(),
            "is_demo": True,
            "errors": [],
        }

    errors: list[str] = []
    asof = _last_trading_day(date.today()).isoformat()
    limited = False
    cooling = (
        _rate_limited_at > 0
        and time.time() - _rate_limited_at < _RATE_LIMIT_COOLDOWN
    )
    if cooling:
        left = _RATE_LIMIT_COOLDOWN - (time.time() - _rate_limited_at)
        errors.append(f"东财主站限流冷却中（剩余 {left:.0f} 秒），改走延迟行情主机")

    if ak is not None and not cooling:
        try:
            raw = _call_with_timeout(
                ak.stock_sector_fund_flow_rank, 20,
                indicator=period, sector_type="行业资金流"
            )
            frame = _normalise_ak_flow(raw, period)
            if len(frame) >= 20:
                if period == "5日":
                    _save_5d_snapshot(frame, asof)
                meta = {
                    "source": "AkShare / 东方财富",
                    "asof": asof,
                    "is_demo": False,
                    "errors": errors,
                }
                if period == "5日":
                    meta["days"] = 5
                return frame, meta
            errors.append(f"AkShare: 返回行数不足（{len(frame)}）")
        except Exception as exc:
            errors.append(f"AkShare: {_clip_error(exc)}")
            if _is_rate_limited(exc):
                limited = True
                _rate_limited_at = time.time()

    # 限流/冷却时只打延迟主机，既避免加重 push2 限流，又能正常出数
    direct_hosts = [EM_FLOW_DELAY_HOST] if (limited or cooling) else None
    try:
        frame = _fetch_flow_direct(period, hosts=direct_hosts)
        if len(frame) >= 20:
            if period == "5日":
                _save_5d_snapshot(frame, asof)
            meta = {
                "source": "东方财富延迟主机" if direct_hosts else "东方财富直连",
                "asof": asof,
                "is_demo": False,
                "errors": errors,
            }
            if period == "5日":
                meta["days"] = 5
            return frame, meta
        errors.append(f"东方财富直连: 返回行数不足（{len(frame)}）")
    except Exception as exc:
        errors.append(f"东方财富直连: {_clip_error(exc)}")
        if _is_rate_limited(exc):
            _rate_limited_at = time.time()

    return _flow_fallback(period, errors)


def _sw_frame(frame: pd.DataFrame) -> pd.DataFrame:
    filtered = frame[frame["行业"].isin(SW_LEVEL1)].copy()
    if len(filtered) >= 20:
        return filtered
    return frame.copy()


# ---- 市场宽度（涨停/跌停家数，全A本地统计）----
BREADTH_FALLBACK = {
    "limit_up": 48, "limit_down": 12, "up": 2380, "down": 2650,
    "median": -0.21, "asof": "2026-11-05",
}


@st.cache_data(ttl=300, show_spinner=False)
def load_breadth(refresh_token: int, mode: str) -> dict[str, Any]:
    """涨停≈涨幅≥9.9%、跌停≈跌幅≤-9.9%（未区分20cm板块，近似值）。"""
    if mode == "demo" or ak is None:
        return {**BREADTH_FALLBACK, "source": "演示数据" if mode == "demo" else "快照"}
    try:
        df = _call_with_timeout(ak.stock_zh_a_spot_em, 25)
        chg = pd.to_numeric(df["涨跌幅"], errors="coerce").dropna()
        if chg.empty:
            raise ValueError("empty")
        return {
            "limit_up": int((chg >= 9.9).sum()),
            "limit_down": int((chg <= -9.9).sum()),
            "up": int((chg > 0).sum()),
            "down": int((chg < 0).sum()),
            "median": round(float(chg.median()), 2),
            "asof": date.today().isoformat(),
            "source": "东财全A",
        }
    except Exception:
        return {**BREADTH_FALLBACK, "source": "快照兜底"}


# ---- ETF 溢价率（东财 IOPV；取不到则显示"溢价 —"，绝不阻塞页面）----
# 估值指数 → 场内 ETF 映射；没有合适 ETF 的指数不显示溢价
VALUATION_ETF = {
    "科创50": "588000", "科创芯片": "588200", "通信": "515880",
    "恒生科技": "513180", "纳指100": "513100", "红利指数": "515080",
    "新能源": "516160", "医疗": "512170", "医药生物": "512010",
    "有色金属": "512400", "券商": "512000", "银行": "512800",
    "白酒": "512690", "食品饮料": "515170", "煤炭": "515220",
    "军工": "512660",
}
# 资金流面板下方展示的 ETF
FLOW_ETFS = {
    "512480": "半导体ETF", "512170": "医疗ETF", "512000": "券商ETF",
    "512400": "有色ETF", "512660": "军工ETF", "515220": "煤炭ETF",
}
ETF_ULIST_URL = "https://push2delay.eastmoney.com/api/qt/ulist.np/get"


def _num(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result else None


def _etf_secid(code: str) -> str:
    return ("1." if code[:1] in {"5", "6"} else "0.") + code


@st.cache_data(ttl=300, show_spinner=False)
def load_etf_premium(refresh_token: int, mode: str) -> dict[str, dict[str, Any]]:
    """ETF 实时溢价率 = 现价 / IOPV - 1。失败返回空字典，标签显示"溢价 —"。"""
    if mode == "demo" or ak is None:
        return {}
    codes = list(dict.fromkeys(list(VALUATION_ETF.values()) + list(FLOW_ETFS)))

    # 1) 东财单次请求，只取需要的 ETF（最快，失败不影响页面）
    try:
        payload = _json_get(
            ETF_ULIST_URL,
            {
                "fltt": "2", "invt": "2",
                "fields": "f12,f14,f2,f441",
                "secids": ",".join(_etf_secid(code) for code in codes),
                "ut": "bd1d9ddb04089700cf9c27f6f7426281",
            },
            timeout=12,
        )
        diff = ((payload or {}).get("data") or {}).get("diff") or []
        out: dict[str, dict[str, Any]] = {}
        for item in diff:
            code = str(item.get("f12", "")).zfill(6)
            price, iopv = _num(item.get("f2")), _num(item.get("f441"))
            if price is None or iopv is None or iopv == 0:
                continue
            out[code] = {
                "name": str(item.get("f14", "")),
                "premium": round((price / iopv - 1) * 100, 2),
            }
        if out:
            return out
    except Exception:
        pass

    # 2) AkShare 全量 ETF 列表兜底
    try:
        df = _call_with_timeout(ak.fund_etf_spot_em, 20)
    except Exception:
        return {}
    if df is None or df.empty or "代码" not in df.columns or "最新价" not in df.columns:
        return {}
    iopv_col = next(
        (c for c in df.columns if "IOPV" in str(c).upper() or "参考净值" in str(c)), None
    )
    if iopv_col is None:
        return {}
    out = {}
    for _, row in df.iterrows():
        code = str(row["代码"]).zfill(6)
        if code not in codes:
            continue
        price = _num(row["最新价"])
        iopv = _num(row[iopv_col])
        if price is None or iopv is None or iopv == 0:
            continue
        out[code] = {"name": str(row.get("名称", "")),
                     "premium": round((price / iopv - 1) * 100, 2)}
    return out


def _premium_tag(premium: dict[str, Any] | None) -> str:
    if premium is None:
        return '<span class="etf-tag na" title="ETF 实时行情未取到（接口限流或非交易时段）">溢价 —</span>'
    value = float(premium["premium"])
    if value > 1.5:
        return (
            f'<span class="etf-tag hot" title="{premium.get("name", "")}：溢价过高，勿追，建议买场外">'
            f'溢价 {value:+.2f}%</span>'
        )
    if value < -1.0:
        return (
            f'<span class="etf-tag cool" title="{premium.get("name", "")}：折价，有套利空间（注意QDII申赎限制）">'
            f'折价 {value:.2f}%</span>'
        )
    return (
        f'<span class="etf-tag flat" title="{premium.get("name", "")}：溢价正常，可场内交易">'
        f'溢价 {value:+.2f}%</span>'
    )


# ---- 全市场两融余额（杠杆水位）----
MARGIN_CACHE_FILE = DATA_DIR / "margin_history.json"


def _margin_points_from_series(stamps, values) -> list[dict[str, Any]]:
    frame = pd.DataFrame({"date": pd.to_datetime(stamps, errors="coerce"),
                          "total": pd.to_numeric(values, errors="coerce")})
    frame = frame.dropna().sort_values("date")
    if frame.empty:
        raise ValueError("两融序列为空")
    scale = 1e8 if float(frame["total"].median()) > 1e10 else 1.0
    return [
        {"date": stamp.strftime("%Y-%m-%d"), "total": round(float(value) / scale, 2)}
        for stamp, value in zip(frame["date"], frame["total"])
    ]


def _fetch_margin_series() -> tuple[list[dict[str, Any]], str]:
    """沪深两市两融余额（金十/交易所汇总），一次拿全历史，取最近3个月。"""
    if ak is None:
        raise ValueError("akshare 未安装")
    sh = _call_with_timeout(ak.macro_china_market_margin_sh, 15)
    sz = _call_with_timeout(ak.macro_china_market_margin_sz, 15)
    merged = pd.merge(
        sh[["日期", "融资融券余额"]], sz[["日期", "融资融券余额"]],
        on="日期", how="inner", suffixes=("_sh", "_sz"),
    )
    total = pd.to_numeric(merged["融资融券余额_sh"], errors="coerce") + pd.to_numeric(
        merged["融资融券余额_sz"], errors="coerce")
    points = _margin_points_from_series(merged["日期"], total)
    return points, "沪深两融"


def _fetch_margin_sse() -> tuple[list[dict[str, Any]], str]:
    """兜底：上交所官方两融汇总（仅沪市）。"""
    if ak is None:
        raise ValueError("akshare 未安装")
    df = _call_with_timeout(
        ak.stock_margin_sse, 15,
        start_date=(date.today() - timedelta(days=120)).strftime("%Y%m%d"),
        end_date=date.today().strftime("%Y%m%d"),
    )
    points = _margin_points_from_series(df["信用交易日期"], df["融资融券余额"])
    return points, "上交所两融（仅沪市）"


def _save_margin_cache(points: list[dict[str, Any]]) -> None:
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        MARGIN_CACHE_FILE.write_text(
            json.dumps({"asof": points[-1]["date"], "points": points}, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError:
        pass


def _load_margin_cache() -> list[dict[str, Any]]:
    if not MARGIN_CACHE_FILE.exists():
        return []
    try:
        payload = json.loads(MARGIN_CACHE_FILE.read_text(encoding="utf-8"))
        points = [
            {"date": str(item["date"]), "total": float(item["total"])}
            for item in payload.get("points", [])
            if item.get("date") and item.get("total") is not None
        ]
    except Exception:
        return []
    return points if len(points) >= 20 else []


def _demo_margin_points() -> list[dict[str, Any]]:
    days = pd.bdate_range(end=_last_trading_day(date.today()), periods=62)
    points = []
    for index, stamp in enumerate(days):
        drift = -0.0016 * index if index < 52 else 0.0
        noise = ((index * 7) % 11 - 5) * 2.2
        points.append({"date": stamp.date().isoformat(),
                       "total": round(15200 * (1 + drift) + noise, 2)})
    return points


@st.cache_data(ttl=86400, show_spinner=False)
def load_margin(
    refresh_token: int, mode: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """两融余额每天更新一次即可，缓存 1 天；失败退回本地真实快照，不造假。"""
    if mode == "demo":
        points = _demo_margin_points()
        return points, {"source": "演示数据", "is_demo": True,
                        "asof": points[-1]["date"], "stale": False}
    errors: list[str] = []
    points: list[dict[str, Any]] = []
    source = ""
    for fetcher, label in ((_fetch_margin_series, "沪深两融"), (_fetch_margin_sse, "上交所两融")):
        try:
            points, source = fetcher()
        except Exception as exc:
            errors.append(f"{label}: {_clip_error(exc)}")
        if len(points) >= 20:
            break
    cutoff = (date.today() - timedelta(days=120)).isoformat()
    recent = [p for p in points if p["date"] >= cutoff][-62:]
    if len(recent) >= 20:
        _save_margin_cache(recent)
        return recent, {"source": source, "is_demo": False,
                        "asof": recent[-1]["date"], "stale": False}
    cached = _load_margin_cache()
    if cached:
        return cached, {"source": "本地两融快照", "is_demo": False,
                        "asof": cached[-1]["date"], "stale": True}
    return [], {"source": "两融数据暂不可用", "is_demo": False, "asof": "", "stale": False}


def margin_signal(points: list[dict[str, Any]]) -> dict[str, str]:
    if len(points) < 10:
        return {"level": "neutral", "label": "样本不足", "detail": "两融数据还在积累，先看资金与估值。"}
    values = [float(p["total"]) for p in points]
    latest = values[-1]
    chg5 = (latest / values[-6] - 1) * 100 if values[-6] else 0.0
    base20 = values[-21] if len(values) >= 21 else values[0]
    chg20 = (latest / base20 - 1) * 100 if base20 else 0.0
    streak = 0
    for index in range(len(values) - 1, 0, -1):
        if values[index] < values[index - 1]:
            streak += 1
        else:
            break
    if (chg5 <= -1.0 and streak >= 3) or (chg20 <= -3.0 and chg5 <= -0.5):
        streak_txt = ""
        if streak >= 3:
            streak_txt = f"连降 {streak} 日、" if streak <= 10 else "连降 10+ 日、"
        return {"level": "warn", "label": "杠杆正在出清",
                "detail": f"两融{streak_txt}近5日 {chg5:+.1f}%、近20日 {chg20:+.1f}%：去杠杆未止，别急着抄底。"}
    if chg5 >= 1.0:
        return {"level": "neutral", "label": "杠杆资金回补",
                "detail": f"近5日 {chg5:+.1f}%（20日 {chg20:+.1f}%）：杠杆情绪回升，别在亢奋期追高。"}
    if chg20 <= -3.0 and chg5 > -0.5:
        return {"level": "good", "label": "出清接近尾声",
                "detail": f"近20日已回落 {chg20:+.1f}%，近5日止跌 {chg5:+.1f}%：等资金回流确认再动手。"}
    if chg5 <= -0.5:
        return {"level": "warn", "label": "杠杆小幅回落",
                "detail": f"近5日 {chg5:+.1f}%：抛压不大但未出清，继续观察。"}
    return {"level": "neutral", "label": "杠杆水位平稳",
            "detail": f"近5日 {chg5:+.1f}% · 近20日 {chg20:+.1f}%。"}


def margin_alert(points: list[dict[str, Any]]) -> dict[str, str]:
    """两融余额 + 5日变化率 → 五档定性预警（红/黄/绿/蓝）。"""
    if len(points) < 6:
        return {"level": "gray", "text": "两融样本不足，暂不做杠杆定性判断。"}
    values = [float(p["total"]) for p in points]
    latest = values[-1]
    chg5 = (latest / values[-6] - 1) * 100 if values[-6] else 0.0
    window = values[-5:]
    mean5 = sum(window) / len(window)
    swing5 = (max(window) - min(window)) / mean5 * 100 if mean5 else 99.0

    if latest > 28000 and chg5 > 0:
        return {"level": "red",
                "text": "⚠️ 极度危险：杠杆资金极度拥挤且仍在狂奔，随时可能引发踩踏，严禁开新仓！"}
    if latest < 18000:
        return {"level": "green",
                "text": "🟢 绝对地量：杠杆泡沫挤干，中长线黄金坑，逢低分批建仓。"}
    if latest < 20000 and chg5 > 2:
        return {"level": "blue",
                "text": "📈 资金入场：杠杆资金开始试探性买入，情绪回暖，可跟随主力做多。"}
    if 23000 <= latest <= 28000 and chg5 < -2:
        return {"level": "yellow",
                "text": "⚙️ 杠杆正在出清：资金被动平仓踩踏中，左侧勿接飞刀，死等企稳。"}
    if -1 <= chg5 <= 1 and swing5 <= 0.25:
        return {"level": "green",
                "text": "✅ 杠杆出清尾声：资金抛压衰竭，底部特征明显，可开始寻找右侧机会。"}
    return {"level": "gray",
            "text": f"杠杆水位中性（5日{chg5:+.2f}%），未见极端信号，按既有纪律执行。"}


# ---- 宽基ETF异动（国家队护盘/建仓痕迹）----
BROAD_ETF_UNIVERSE = [
    {"name": "沪深300ETF", "code": "510300", "short": "沪深300"},
    {"name": "上证50ETF", "code": "510050", "short": "上证50"},
    {"name": "科创50ETF", "code": "588000", "short": "科创50"},
]


def _etf_anomaly_frame(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.copy()
    frame.columns = [str(col) for col in frame.columns]
    frame["日期"] = pd.to_datetime(frame["日期"], errors="coerce")
    frame["成交额"] = pd.to_numeric(frame["成交额"], errors="coerce")
    frame["涨跌幅"] = pd.to_numeric(frame["涨跌幅"], errors="coerce")
    frame = frame.dropna(subset=["日期", "成交额"]).sort_values("日期")
    frame["均额20"] = frame["成交额"].rolling(window=20).mean()
    frame["放量倍数"] = frame["成交额"] / frame["均额20"]
    frame["异动"] = (frame["放量倍数"] >= 1.5) & (frame["涨跌幅"] > 0)
    return frame.dropna(subset=["均额20"]).reset_index(drop=True)


def _etf_secid(code: str) -> str:
    return ("1." if code.startswith(("5", "6", "9")) else "0.") + code


def _fetch_etf_kline_em(code: str) -> pd.DataFrame:
    """东方财富 K 线直连：f51 日期 / f53 收盘 / f57 成交额 / f59 涨跌幅。"""
    params = {
        "secid": _etf_secid(code),
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "klt": "101",
        "fqt": "1",
        "beg": (date.today() - timedelta(days=200)).strftime("%Y%m%d"),
        "end": date.today().strftime("%Y%m%d"),
        "lmt": "120",
    }
    data = _json_get(EM_KLINE_URL, params, timeout=12)
    klines = ((data or {}).get("data") or {}).get("klines") or []
    rows = []
    for item in klines:
        parts = str(item).split(",")
        if len(parts) < 9:
            continue
        rows.append({"日期": parts[0], "收盘": parts[2], "成交额": parts[6], "涨跌幅": parts[8]})
    return pd.DataFrame(rows)


def _fetch_etf_kline_tencent(code: str) -> pd.DataFrame:
    """腾讯日线兜底（云端访问海外链路更稳）；成交额用 收盘价×成交量(手)×100 估算。"""
    symbol = ("sh" if code.startswith(("5", "6", "9")) else "sz") + code
    response = requests.get(
        "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get",
        params={"param": f"{symbol},day,,,120,qfq"},
        headers=HTTP_HEADERS,
        timeout=12,
    )
    response.raise_for_status()
    payload = response.json().get("data", {}).get(symbol, {})
    series = payload.get("qfqday") or payload.get("day") or []
    rows = []
    prev_close = None
    for item in series:
        if len(item) < 6:
            continue
        close = float(item[2])
        volume = float(item[5])
        pct = (close / prev_close - 1) * 100 if prev_close else 0.0
        rows.append({"日期": item[0], "收盘": close, "成交额": close * volume * 100, "涨跌幅": pct})
        prev_close = close
    return pd.DataFrame(rows)


def _fetch_etf_kline_sina(code: str) -> pd.DataFrame:
    """新浪日线兜底（akshare 封装），成交额用 收盘价×成交量 估算。"""
    if ak is None:
        raise RuntimeError("AkShare 未安装")
    symbol = ("sh" if code.startswith(("5", "6", "9")) else "sz") + code
    raw = _call_with_timeout(ak.fund_etf_hist_sina, 15, symbol=symbol)
    frame = raw.rename(columns={"date": "日期", "close": "收盘", "volume": "成交量"})
    frame["成交额"] = pd.to_numeric(frame["收盘"], errors="coerce") * pd.to_numeric(
        frame["成交量"], errors="coerce"
    )
    frame["涨跌幅"] = pd.to_numeric(frame["收盘"], errors="coerce").pct_change() * 100
    return frame[["日期", "收盘", "成交额", "涨跌幅"]]


@st.cache_data(ttl=86400, show_spinner=False)
def _etf_hist_cached(code: str, refresh_token: int) -> tuple[pd.DataFrame, str]:
    """日线多源兜底。失败时抛异常而不是返回空表，避免把超时结果缓存一整天。"""
    errors: list[str] = []

    if ak is not None:
        try:
            raw = _call_with_timeout(
                ak.fund_etf_hist_em, 25,
                symbol=code, period="daily", adjust="qfq",
                start_date=(date.today() - timedelta(days=200)).strftime("%Y%m%d"),
                end_date=date.today().strftime("%Y%m%d"),
            )
            if raw is None or len(raw) < 30:
                raise ValueError(f"样本不足（{0 if raw is None else len(raw)} 条）")
            return raw, "东方财富ETF日线"
        except Exception as exc:
            errors.append(f"东财AkShare: {_clip_error(exc, 70)}")

    for label, fetcher in (
        ("东财直连", _fetch_etf_kline_em),
        ("腾讯日线", _fetch_etf_kline_tencent),
        ("新浪日线", _fetch_etf_kline_sina),
    ):
        try:
            raw = fetcher(code)
            if raw is None or len(raw) < 30:
                raise ValueError(f"样本不足（{0 if raw is None else len(raw)} 条）")
            return raw, label
        except Exception as exc:
            errors.append(f"{label}: {_clip_error(exc, 60)}")

    raise RuntimeError("；".join(errors)[:180] or "全部数据源不可用")


def load_etf_anomaly(
    code: str, refresh_token: int, mode: str
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """宽基ETF近30个交易日量价异动：放量≥1.5倍且收红。日线数据缓存1天。"""
    if mode == "demo":
        days = pd.bdate_range(end=_last_trading_day(date.today()), periods=60)
        rng = pd.Series([1 + ((i * 13) % 17) / 100 for i in range(len(days))])
        amounts = (18e8 * rng * (1 + 0.02 * pd.Series(range(len(days))).mod(7))).tolist()
        for idx in (24, 41, 52):
            amounts[idx] *= 1.9
        frame = _etf_anomaly_frame(
            pd.DataFrame({
                "日期": days,
                "成交额": amounts,
                "涨跌幅": [0.8 if i % 5 else -0.6 for i in range(len(days))],
            })
        )
        return frame.tail(30).reset_index(drop=True), {
            "source": "演示数据", "is_demo": True,
            "asof": frame["日期"].iloc[-1].strftime("%Y-%m-%d"), "errors": [],
        }

    try:
        raw, source = _etf_hist_cached(code, refresh_token)
        frame = _etf_anomaly_frame(raw)
        if len(frame) < 30:
            raise ValueError(f"可用样本不足（{len(frame)} 条）")
        return frame.tail(30).reset_index(drop=True), {
            "source": source, "is_demo": False,
            "asof": frame["日期"].iloc[-1].strftime("%Y-%m-%d"), "errors": [],
        }
    except Exception as exc:
        try:
            _etf_hist_cached.clear(code, refresh_token)
        except Exception:
            pass
        return pd.DataFrame(), {"source": "ETF日线暂不可用", "is_demo": False,
                                "asof": "", "errors": [_clip_error(exc)]}


def etf_anomaly_signal(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"count": 0, "last": "", "label": "数据暂不可用", "level": "neutral"}
    hits = frame[frame["异动"]]
    last = hits["日期"].iloc[-1].strftime("%m-%d") if not hits.empty else ""
    if hits.empty:
        return {"count": 0, "last": "", "level": "neutral",
                "label": "近30日无放量收红，暂未看到大资金护盘痕迹"}
    recent = hits.tail(3)
    biggest = recent["放量倍数"].max()
    dates = "、".join(hits.tail(4)["日期"].dt.strftime("%m-%d"))
    return {
        "count": len(hits), "last": last, "level": "warn",
        "label": f"近30日异动 {len(hits)} 次（{dates}），最近 {last}；最大放量 {biggest:.1f} 倍",
    }


def _flow_extremes(frame: pd.DataFrame, count: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = frame.dropna(subset=["净流入"]).drop_duplicates(subset=["行业"])
    inflow = frame.nlargest(count, "净流入")
    outflow = frame.nsmallest(count, "净流入")
    return inflow, outflow


def _flow_card_lines(
    frame: pd.DataFrame,
    avg5_map: dict[str, float] | None = None,
    avg_label: str = "5日均",
) -> str:
    """行业列表文案；传入 5日累计额时，附上日均值做参考。"""
    lines: list[str] = []
    for _, row in frame.iterrows():
        name = str(row["行业"])
        value = float(row["净流入"]) / 1e8
        line = f"{name}  {value:+.2f}亿"
        if avg5_map and name in avg5_map:
            line += f' <span class="favg">{avg_label} {avg5_map[name] / 1e8:+.2f}亿</span>'
        lines.append(line)
    return "<br>".join(lines)


def _load_history_file() -> list[dict[str, Any]]:
    if not FLOW_HISTORY_FILE.exists():
        return []
    try:
        payload = json.loads(FLOW_HISTORY_FILE.read_text(encoding="utf-8"))
        snapshots = payload.get("snapshots", []) if isinstance(payload, dict) else payload
        if isinstance(snapshots, list):
            return snapshots
    except Exception:
        return []
    return []


def _record_flow_snapshot(frame: pd.DataFrame, meta: dict[str, Any]) -> list[dict[str, Any]]:
    history = _load_history_file()
    if meta.get("is_demo"):
        return history

    snapshot_date = _last_trading_day(date.today()).isoformat()
    flows = {
        str(row["行业"]): float(row["净流入"])
        for _, row in _sw_frame(frame).iterrows()
        if pd.notna(row["净流入"])
    }
    entry = {"date": snapshot_date, "flows": flows}
    history = [item for item in history if item.get("date") != snapshot_date]
    history.append(entry)
    history = sorted(history, key=lambda item: str(item.get("date", "")))[-15:]

    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        FLOW_HISTORY_FILE.write_text(
            json.dumps({"snapshots": history}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError:
        pass
    return history


def _demo_history() -> list[dict[str, Any]]:
    dates = pd.bdate_range(end=_last_trading_day(date.today()), periods=3)
    history = []
    for index, (stamp, scale) in enumerate(zip(dates, [0.66, 0.82, 1.0])):
        frame = _demo_flow_frame("今日", scale=scale, jitter=0.04 * index)
        history.append(
            {
                "date": stamp.date().isoformat(),
                "flows": {row["行业"]: float(row["净流入"]) for _, row in frame.iterrows()},
            }
        )
    return history


def get_flow_history(
    today_frame: pd.DataFrame, today_meta: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if today_meta.get("is_demo"):
        return _demo_history(), {
            "source": "演示三日样本",
            "is_demo": True,
        }
    history = _record_flow_snapshot(today_frame, today_meta)
    return history, {
        "source": f"本地快照 {len(history)} 日",
        "is_demo": False,
    }


def detect_rotation(history: list[dict[str, Any]]) -> dict[str, Any]:
    sorted_history = sorted(history, key=lambda item: str(item.get("date", "")))
    recent = sorted_history[-3:]
    if len(recent) < 3:
        return {
            "triggered": False,
            "ready": False,
            "days": len(recent),
            "tech_out": False,
            "low_top": [],
            "message": f"历史样本 {len(recent)}/3 天，高低切预警等待数据积累。",
        }

    tech_out = all(
        float(snapshot.get("flows", {}).get(name, 0)) < 0
        for snapshot in recent
        for name in TECH_FLOW_NAMES
    )

    low_top_sets: list[set[str]] = []
    for snapshot in recent:
        flows = snapshot.get("flows", {})
        market_top = sorted(
            [
                (name, float(value))
                for name, value in flows.items()
                if name in SW_LEVEL1 and float(value) > 0
            ],
            key=lambda item: item[1],
            reverse=True,
        )[:3]
        low_top_sets.append(
            {name for name, _ in market_top if name in LOW_FLOW_NAMES}
        )

    common_low = set.intersection(*low_top_sets) if low_top_sets else set()
    triggered = bool(tech_out and common_low)
    low_top = sorted(common_low)
    if triggered:
        message = (
            "资金高低切预警：警惕科技资金抽血，关注接力方向"
            f"（{'、'.join(low_top)}）。"
        )
    else:
        message = "未触发高低切：需科技（电子/通信）连续 3 日净流出，且低位板块连续 3 日流入前三。"
    return {
        "triggered": triggered,
        "ready": True,
        "days": 3,
        "tech_out": tech_out,
        "low_top": low_top,
        "message": message,
    }


def find_opportunities(
    history: list[dict[str, Any]], valuation_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], str]:
    sorted_history = sorted(history, key=lambda item: str(item.get("date", "")))
    if len(sorted_history) < 3:
        return [], f"三日样本不足（{len(sorted_history)}/3），暂不生成机会雷达。"

    recent = sorted_history[-3:]
    valuation_map = {row["name"]: row for row in valuation_rows}
    flow_to_valuation = {
        "电力设备": "电力设备",
        "有色金属": "有色金属",
        "医药生物": FLOW_VALUATION_ALIAS.get("医药生物", "医药生物"),
        "公用事业": None,
        "基础化工": None,
    }

    top_sets = []
    for snapshot in recent:
        flows = snapshot.get("flows", {})
        ranked = sorted(
            [
                (name, float(value))
                for name, value in flows.items()
                if name in SW_LEVEL1 and float(value) > 0
            ],
            key=lambda item: item[1],
            reverse=True,
        )[:3]
        top_sets.append({name for name, _ in ranked})

    candidates = []
    for flow_name, value_name in flow_to_valuation.items():
        if value_name is None:
            continue
        if not all(flow_name in top_set for top_set in top_sets):
            continue
        if not all(float(item.get("flows", {}).get(flow_name, 0)) > 0 for item in recent):
            continue
        valuation = valuation_map.get(value_name)
        if not valuation or float(valuation["pct"]) >= 30:
            continue
        three_day = sum(float(item.get("flows", {}).get(flow_name, 0)) for item in recent)
        candidates.append(
            {
                "name": value_name,
                "flow_name": flow_name,
                "pct": float(valuation["pct"]),
                "pe": float(valuation["pe"]),
                "three_day": three_day,
            }
        )

    candidates.sort(key=lambda item: item["pct"])
    if not candidates:
        return [], "暂无同时满足“连续 3 日流入 + PE 分位<30%”的非科技板块。"
    return candidates, ""


# 事件 → 板块 关键词映射：用于“资金×事件”共振判断
EVENT_SECTOR_KEYWORDS = {
    "半导体": ["半导体", "芯片", "中芯", "英伟达", "集成电路", "出口管制", "AI", "算力"],
    "通信": ["通信", "6G", "5G", "光模块", "AI", "算力"],
    "新能源": ["新能源", "光伏", "储能", "锂电", "电池", "电动车"],
    "电力设备": ["电力", "电网", "储能", "光伏", "新能源", "特高压"],
    "医疗": ["医疗", "医药", "创新药", "医保", "集采", "医疗器械"],
    "医药生物": ["医疗", "医药", "创新药", "医保", "集采", "医疗"],
    "有色金属": ["有色", "稀土", "黄金", "铜", "铝", "锂"],
}
TECH_EVENT_KEYWORDS = [
    "半导体", "芯片", "中芯", "英伟达", "AI", "听证", "出口管制", "6G", "算力", "科技",
]


def find_resonance(
    events: list[dict[str, Any]],
    opportunities: list[dict[str, Any]],
    flow_frame: pd.DataFrame,
) -> dict[str, Any]:
    """资金×事件共振：科技流出+科技利空临近=双重利空；低位流入+相关催化临近=共振机会。"""
    flows = {
        str(row["行业"]): float(row["净流入"])
        for _, row in flow_frame.iterrows()
        if pd.notna(row["净流入"])
    }
    tech_out = any(flows.get(name, 0) < 0 for name in TECH_FLOW_NAMES)
    soon = [event for event in events if 0 <= _event_days(event) <= 7]

    negatives: list[str] = []
    positives: list[str] = []
    for event in soon:
        title = str(event["title"])
        if tech_out and any(k in title for k in TECH_EVENT_KEYWORDS):
            negatives.append(title)
        for item in opportunities:
            keywords = EVENT_SECTOR_KEYWORDS.get(item["name"], [])
            if any(k in title for k in keywords):
                positives.append(f"{item['name']} ↔ {title}")
    return {"negatives": negatives, "positives": positives}


# ---------------------------------------------------------------------------
# 单日高低切异动 + 连续流入雷达（V2.0）
# ---------------------------------------------------------------------------

TECH_BROAD = ["电子", "通信", "计算机"]
LOW_BROAD = ["医药生物", "有色金属", "电力设备", "非银金融", "食品饮料"]


def detect_daily_rotation(sw_frame: pd.DataFrame) -> dict[str, Any]:
    """单日口径：科技在流出前五 + 低位在流入前五 → 黄灯异动。"""
    if sw_frame.empty:
        return {"triggered": False, "text": ""}
    inflow_names = sw_frame[sw_frame["净流入"] > 0].nlargest(5, "净流入")["行业"].tolist()
    outflow_names = sw_frame[sw_frame["净流入"] < 0].nsmallest(5, "净流入")["行业"].tolist()
    tech_out = [n for n in outflow_names if n in TECH_BROAD]
    low_in = [n for n in inflow_names if n in LOW_BROAD]
    if tech_out and low_in:
        return {
            "triggered": True,
            "text": f"单日异动：{'、'.join(tech_out)}遭抽血，{'、'.join(low_in)}流入居前——关注低位接力方向。",
        }
    return {"triggered": False, "text": ""}


def find_consecutive_inflows(
    history: list[dict[str, Any]], valuation_rows: list[dict[str, Any]], top_n: int = 3
) -> list[dict[str, Any]]:
    """连续流入雷达：近3日每日净流入为正的行业，按3日累计排序取前N。"""
    sorted_history = sorted(history, key=lambda item: str(item.get("date", "")))
    if len(sorted_history) < 3:
        return []
    recent = sorted_history[-3:]
    valuation_map = {row["name"]: row for row in valuation_rows}
    result = []
    for name in SW_LEVEL1:
        daily = [float(item.get("flows", {}).get(name, 0) or 0) for item in recent]
        if all(v > 0 for v in daily):
            alias = FLOW_VALUATION_ALIAS.get(name, name)
            v = valuation_map.get(alias)
            result.append({
                "name": alias,
                "flow_name": name,
                "total": sum(daily),
                "pct": float(v["pct"]) if v else None,
            })
    result.sort(key=lambda x: x["total"], reverse=True)
    return result[:top_n]


# ---------------------------------------------------------------------------
# 跨市场宏观联动（V2.0）
# ---------------------------------------------------------------------------

SINA_QUOTE_URL = "https://hq.sinajs.cn/list={codes}"
SINA_HEADERS = {**HTTP_HEADERS, "Referer": "https://finance.sina.com.cn/"}

# 演示/兜底快照（2026-11-06 新浪 & 中债数据）
MACRO_FALLBACK = {
    "dxy":    {"name": "美元指数", "value": 101.90, "chg": 0.30, "prev": 101.59},
    "us10y":  {"name": "美债10Y收益率", "value": 4.84, "chg": -1.22, "prev": 4.90},
    "usdcnh": {"name": "美元/离岸人民币", "value": 6.6664, "chg": 0.02, "pips": 16.0, "prev": 6.6648},
    "gold":   {"name": "COMEX黄金", "value": 4054.5, "chg": -1.43, "prev": 4113.5},
    "tips":   {"name": "美债实际利率TIPS", "value": 1.92, "chg": -1.03, "prev": 1.94},
    "south":  {"name": "南向资金净买额", "value": -213.26, "chg": 0.0, "asof": "2026-11-05"},
}


def _parse_sina_line(text: str, code: str) -> list[str]:
    m = re.search(rf'hq_str_{code}="([^"]*)"', text)
    if not m or not m.group(1).strip():
        raise ValueError(f"{code} 无数据")
    return m.group(1).split(",")


def _fetch_sina_macro() -> dict[str, Any]:
    resp = requests.get(
        SINA_QUOTE_URL.format(codes="DINIW,fx_susdcnh,hf_GC"),
        headers=SINA_HEADERS, timeout=12,
    )
    resp.encoding = "gbk"
    text = resp.text

    dxy = _parse_sina_line(text, "DINIW")
    dxy_latest, dxy_prev = float(dxy[1]), float(dxy[3])

    cnh = _parse_sina_line(text, "fx_susdcnh")
    cnh_latest, cnh_prev = float(cnh[1]), float(cnh[3])

    gold = _parse_sina_line(text, "hf_GC")
    gold_latest, gold_prev = float(gold[0]), float(gold[7])

    return {
        "dxy": {"value": dxy_latest, "prev": dxy_prev,
                "chg": (dxy_latest / dxy_prev - 1) * 100},
        "usdcnh": {"value": cnh_latest, "prev": cnh_prev,
                   "chg": (cnh_latest / cnh_prev - 1) * 100,
                   "pips": (cnh_latest - cnh_prev) * 10000},
        "gold": {"value": gold_latest, "prev": gold_prev,
                 "chg": (gold_latest / gold_prev - 1) * 100},
    }


def _fetch_us10y() -> dict[str, Any]:
    """美债10年收益率：中债官网数据（T+1），经 akshare 拉取。"""
    if ak is None:
        raise ValueError("akshare 未安装")
    start = (date.today() - timedelta(days=25)).strftime("%Y%m%d")
    df = _call_with_timeout(ak.bond_zh_us_rate, 15, start_date=start)
    col = "美国国债收益率10年"
    series = pd.to_numeric(df[col], errors="coerce").dropna()
    if len(series) < 2:
        raise ValueError("美债收益率样本不足")
    latest, prev = float(series.iloc[-1]), float(series.iloc[-2])
    return {"value": latest, "prev": prev, "chg": (latest / prev - 1) * 100,
            "asof": str(df["日期"].iloc[-1])}


def _fetch_tips() -> dict[str, Any]:
    """美债10年TIPS实际利率：优先 yfinance，失败转 FRED 免key CSV（DFII10）。"""
    # 1) yfinance（国内网络大概率失败，自动降级）
    try:
        import yfinance as yf
        hist = _call_with_timeout(yf.download, 15, "DFII10", period="1mo", progress=False)
        if hist is not None and not hist.empty:
            close = hist["Close"].dropna()
            if len(close) >= 2:
                latest, prev = float(close.iloc[-1]), float(close.iloc[-2])
                return {"value": round(latest, 2), "prev": prev,
                        "chg": (latest - prev) * 100,  # bp
                        "asof": str(close.index[-1].date())}
    except Exception:
        pass
    # 2) FRED 免 key CSV
    resp = requests.get(
        "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DFII10",
        headers=HTTP_HEADERS, timeout=12,
    )
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    df = df[df["DFII10"].astype(str) != "."]
    series = pd.to_numeric(df["DFII10"], errors="coerce").dropna()
    if len(series) < 2:
        raise ValueError("TIPS 样本不足")
    latest, prev = float(series.iloc[-1]), float(series.iloc[-2])
    return {"value": round(latest, 2), "prev": prev,
            "chg": (latest - prev) * 100,
            "asof": str(df["observation_date"].iloc[-1])}


def _fetch_south() -> dict[str, Any]:
    """南向资金（港股通）当日成交净买额，单位亿元。"""
    if ak is None:
        raise ValueError("akshare 未安装")
    df = _call_with_timeout(ak.stock_hsgt_hist_em, 15, symbol="南向资金")
    if df is None or df.empty:
        raise ValueError("南向数据为空")
    latest = df.iloc[-1]
    return {"value": round(float(latest["当日成交净买额"]), 2),
            "asof": str(latest["日期"]), "chg": 0.0}


@st.cache_data(ttl=1800, show_spinner=False)
def load_macro(refresh_token: int, mode: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if mode == "demo":
        return {k: dict(v) for k, v in MACRO_FALLBACK.items()}, {
            "source": "演示数据", "is_demo": True, "note": ""}

    macro = {k: dict(v) for k, v in MACRO_FALLBACK.items()}
    errors: list[str] = []

    fetchers = {
        "sina": _fetch_sina_macro,
        "us10y": _fetch_us10y,
        "tips": _fetch_tips,
        "south": _fetch_south,
    }
    labels = {"sina": "新浪行情", "us10y": "美债收益率", "tips": "TIPS", "south": "南向资金"}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {key: pool.submit(fn) for key, fn in fetchers.items()}
        results: dict[str, Any] = {}
        for key, future in futures.items():
            try:
                results[key] = future.result()
            except Exception as exc:
                errors.append(f"{labels[key]}: {_clip_error(exc)}")

    live = 0
    if "sina" in results:
        for k, v in results["sina"].items():
            macro[k].update(v)
        live += 3
    for key in ("us10y", "tips", "south"):
        if key in results:
            macro[key] = {**macro[key], **results[key]}
            live += 1

    meta = {
        "source": "实时宏观" if live >= 5 else ("部分实时+快照" if live else "宏观快照兜底"),
        "is_demo": False,
        "note": "美债10Y/TIPS为T+1数据",
        "errors": errors,
    }
    return macro, meta


def macro_signal(macro: dict[str, Any]) -> dict[str, str]:
    """红/黄/绿/中性 四档宏观信号。"""
    dxy, us10y, cnh, gold = macro["dxy"], macro["us10y"], macro["usdcnh"], macro["gold"]
    if cnh.get("pips", 0) >= 300:
        return {
            "level": "red",
            "title": "汇率承压",
            "text": f"离岸人民币单日急贬 {cnh['pips']:.0f} 点（{cnh['chg']:+.2f}%）：外资流出预警，压制A股核心资产与恒生科技。",
        }
    if us10y["chg"] > 2 and dxy["chg"] > 0:
        return {
            "level": "red",
            "title": "宏观流动性收紧",
            "text": "美债收益率单日大涨 + 美元走强：极度利空纳指与A股高估值科技股，今日严禁追高科技。",
        }
    if cnh.get("pips", 0) > 150:
        return {
            "level": "yellow",
            "title": "人民币走弱",
            "text": f"离岸人民币单日贬值 {cnh['pips']:.0f} 点：外资流出预警，大盘蓝筹与核心资产面临抛压。",
        }
    if cnh.get("pips", 0) <= -150:
        return {
            "level": "green",
            "title": "汇率走强",
            "text": f"离岸人民币单日升值 {abs(cnh['pips']):.0f} 点：利好A股/港股与外资重仓的核心资产。",
        }
    if gold["chg"] > 1 and us10y["chg"] < 0:
        return {
            "level": "green",
            "title": "避险模式开启",
            "text": "黄金大涨 + 美债收益率回落：留意持仓中黄金与红利的防御价值。",
        }
    return {"level": "neutral", "title": "宏观中性", "text": "外围未出现极端信号，按资金与估值执行纪律。"}


# ---------------------------------------------------------------------------
# 事件与 AI
# ---------------------------------------------------------------------------


EVENTS_PROMPT_TEMPLATE = """你是一个金融日历助手。今天是 {today}，请列出未来 10-15 天内，全球和中国资本市场即将发生的重要宏观数据（如美国CPI/非农、中国LPR）、重大产业大会（如科技大会、医药大会）以及关键龙头财报（如英伟达、中芯国际）。
只输出纯 JSON 数组，不要 Markdown 代码块、不要任何解释文字。
每个元素包含字段：date（YYYY-MM-DD）、title、impact（一句话说明对A股/科技板块的影响）、description、type（宏观数据/产业催化/龙头财报）。
日期必须落在 {today} 之后 15 天内，按日期升序，最多 12 条。"""


def generate_events(api_key: str, base_url: str, model: str) -> tuple[list[dict[str, Any]], str]:
    """调用 DeepSeek 生成未来 15 天事件日历，成功则覆盖写入 events.json。"""
    if not api_key:
        return [], "未配置 DeepSeek Key，无法自动生成事件日历。"
    today = date.today()
    try:
        content = _chat_completion(
            api_key, base_url, model,
            system="你是严谨的金融日历助手，只输出合法 JSON 数组。",
            user=EVENTS_PROMPT_TEMPLATE.format(today=today.isoformat()),
            temperature=0.2,
            max_tokens=1200,
        )
        events = _parse_events_json(content, today)
        if not events:
            raise ValueError("模型未返回有效事件")
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        EVENTS_FILE.write_text(
            json.dumps(events, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return events, ""
    except Exception as exc:
        return [], _friendly_error(exc)


def _parse_events_json(text: str, today: date) -> list[dict[str, Any]]:
    cleaned = re.sub(r"```(?:json)?", "", str(text)).strip().strip("`")
    start, end = cleaned.find("["), cleaned.rfind("]")
    if start < 0 or end < start:
        raise ValueError("模型返回内容不是 JSON 数组")
    payload = json.loads(cleaned[start:end + 1])
    events: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in payload if isinstance(payload, list) else []:
        if not isinstance(item, dict):
            continue
        day = pd.to_datetime(item.get("date"), errors="coerce")
        title = str(item.get("title") or "").strip()
        if pd.isna(day) or not title:
            continue
        day = day.date()
        if not (today - timedelta(days=1) <= day <= today + timedelta(days=20)):
            continue
        if title in seen:
            continue
        seen.add(title)
        events.append({
            "date": day.isoformat(),
            "title": title,
            "type": str(item.get("type") or "宏观事件"),
            "impact": str(item.get("impact") or item.get("description") or "").strip(),
        })
    events.sort(key=lambda row: row["date"])
    return events[:12]


def _events_stale(days: int = 7) -> bool:
    try:
        age = datetime.now().timestamp() - EVENTS_FILE.stat().st_mtime
    except OSError:
        return True
    return age > days * 86400


def load_events() -> tuple[list[dict[str, Any]], str]:
    if EVENTS_FILE.exists():
        try:
            payload = json.loads(EVENTS_FILE.read_text(encoding="utf-8"))
            if isinstance(payload, list) and payload:
                events = [
                    {
                        "date": str(item["date"]),
                        "title": str(item["title"]),
                        "type": str(item.get("type", "宏观事件")),
                        "impact": str(item.get("impact", "")),
                    }
                    for item in payload
                    if item.get("date") and item.get("title")
                ]
                events.sort(key=lambda item: item["date"])
                return events, "events.json"
        except Exception:
            pass
    return DEFAULT_EVENTS, "代码内静态配置"


def _flow_text(frame: pd.DataFrame, count: int = 5) -> str:
    if frame.empty:
        return "暂无数据"
    return "、".join(
        f"{row['行业']} {_fmt_yi(float(row['净流入']))}" for _, row in frame.head(count).iterrows()
    )


def _support_context(supports: list[dict[str, Any]]) -> str:
    parts = []
    for row in supports:
        ma_text = "站上5日线" if row["above_ma5"] else "在5日线下方"
        parts.append(
            f"{row['name']}现价{row['latest']:.2f}，20日支撑{row['low20']:.2f}，"
            f"距支撑{row['dist']:+.2f}%，{ma_text}"
        )
    return "；".join(parts)


def _ai_flow_line(frame: pd.DataFrame, valuation_rows: list[dict[str, Any]], count: int = 5) -> list[str]:
    """资金流 Top/Bottom 与估值分位合并成一行，供 Prompt 使用。"""
    vmap = {row["name"]: row for row in valuation_rows}
    lines: list[str] = []
    for _, row in frame.head(count).iterrows():
        name = str(row["行业"])
        vrow = vmap.get(FLOW_VALUATION_ALIAS.get(name, name)) or vmap.get(name)
        pe_txt = f"PE分位{vrow['pct']:.0f}%" if vrow else "PE分位—"
        pbp_txt = (
            f"PB分位{vrow['pb_pct']:.0f}%"
            if vrow and vrow.get("pb_pct") is not None
            else "PB分位—"
        )
        lines.append(f"{name}: {float(row['净流入']) / 1e8:+.1f}亿 / {pe_txt} / {pbp_txt}")
    return lines


def _ai_etf_text(etf_anomalies: dict[str, tuple[pd.DataFrame, dict[str, Any]]]) -> list[str]:
    lines = []
    for item in BROAD_ETF_UNIVERSE:
        frame, _meta = etf_anomalies.get(item["code"], (pd.DataFrame(), {}))
        if frame is None or frame.empty:
            lines.append(f"{item['name']}：数据暂不可用")
            continue
        sig = etf_anomaly_signal(frame)
        if not sig["count"]:
            lines.append(f"{item['name']}：近30日未见放量收红")
            continue
        lines.append(f"{item['name']}：{sig['label']}")
    return lines


def build_ai_prompt(
    valuation_rows: list[dict[str, Any]],
    inflow: pd.DataFrame,
    outflow: pd.DataFrame,
    supports: list[dict[str, Any]],
    holdings: list[dict[str, Any]],
    macro: dict[str, Any],
    msignal: dict[str, str],
    consec_inflows: list[dict[str, Any]],
    etf_anomalies: dict[str, tuple[pd.DataFrame, dict[str, Any]]],
    margin_points: list[dict[str, Any]],
    total_capital: float,
    tech_position: float,
    cash: float,
) -> str:
    """把全市场原始表格用 Pandas 压缩成四块结构，再交给大模型。"""
    # ---- 1. 宏观与杠杆 ----
    tips = macro.get("tips") or {}
    tips_txt = (
        f"TIPS实际利率{tips['value']:.2f}%（{tips.get('chg', 0):+.2f}）"
        if tips.get("value") is not None
        else "TIPS实际利率暂缺"
    )
    macro_lines = [
        f"美债10Y {macro['us10y']['value']:.2f}%（日变动{macro['us10y']['chg']:+.2f}%）· {tips_txt}",
        f"美元指数{macro['dxy']['value']:.2f}（{macro['dxy']['chg']:+.2f}%）· "
        f"离岸人民币{macro['usdcnh']['value']:.4f}（{macro['usdcnh'].get('pips', 0):+.0f}点，数值上升=贬值）",
        f"黄金{macro['gold']['value']:.0f}美元（{macro['gold']['chg']:+.2f}%）",
    ]
    if margin_points:
        latest = float(margin_points[-1]["total"])
        base5 = float(margin_points[-6]["total"]) if len(margin_points) >= 6 else float(margin_points[0]["total"])
        base20 = float(margin_points[-21]["total"]) if len(margin_points) >= 21 else float(margin_points[0]["total"])
        chg5 = (latest / base5 - 1) * 100 if base5 else 0.0
        chg20 = (latest / base20 - 1) * 100 if base20 else 0.0
        macro_lines.append(
            f"两融余额{latest / 1e4:,.2f}万亿，5日{chg5:+.1f}% / 20日{chg20:+.1f}%："
            f"{margin_signal(margin_points)['label']}"
        )
    else:
        macro_lines.append("两融余额：数据暂不可用")
    macro_lines.append(f"宏观信号：{msignal['title']}——{msignal['text']}")

    # ---- 2. 科技防线 + 科技资金状态 ----
    defense_lines = _support_context(supports).split("；")
    tech_names = ["电子", "通信", "计算机"]
    tech_flow = []
    for name in tech_names:
        hit = inflow[inflow["行业"] == name]
        if hit.empty:
            hit = outflow[outflow["行业"] == name]
        if not hit.empty:
            tech_flow.append(f"{name} {float(hit.iloc[0]['净流入']) / 1e8:+.1f}亿")
    if tech_flow:
        defense_lines.append("科技资金：" + "、".join(tech_flow))

    # ---- 3. 资金流 × 估值 交叉表 ----
    infl_lines = _ai_flow_line(inflow, valuation_rows)
    outfl_lines = _ai_flow_line(outflow, valuation_rows)
    expensive = [r for r in valuation_rows if r["pct"] >= 80]
    cheap = [r for r in valuation_rows if r["pct"] < 20]
    exp_txt = "、".join(f"{r['name']}(PE分位{r['pct']:.0f}%)" for r in expensive) or "无"
    cheap_txt = "、".join(f"{r['name']}(PE分位{r['pct']:.0f}%)" for r in cheap) or "无"
    cheap_names = {r["name"] for r in cheap}
    inflow_names = {
        FLOW_VALUATION_ALIAS.get(str(row["行业"]), str(row["行业"])) for _, row in inflow.iterrows()
    }
    cheap_inflow = [
        f"{c['name']}(PE分位{c['pct']:.0f}%、3日流入{_fmt_compact_yi(c['total'])})"
        for c in consec_inflows
        if c["pct"] is not None and c["pct"] < 20
    ]
    if not cheap_inflow:
        cheap_inflow = [
            item for item in cheap_txt.split("、")
            if item.split("(")[0] in inflow_names
        ]
    if not cheap_inflow:
        cheap_inflow = ["无（低估与资金流入未同时出现）"]

    holding_txt = (
        "；".join(
            f"{h['name']} 成本{h['cost']:.0f}→现价{h['latest']:.0f}，盈亏{h['pnl_pct']:+.1f}%"
            for h in holdings
        )
        if holdings
        else "未录入具体持仓"
    )

    return f"""【Context】
1. 宏观与杠杆：
   - {macro_lines[0]}
   - {macro_lines[1]}
   - {macro_lines[2]}
   - {macro_lines[3] if len(macro_lines) > 3 else "两融余额：数据暂不可用"}
   - 宏观信号：{msignal['title']}——{msignal['text']}
2. ETF国家队动向（近30个交易日，放量≥1.5倍且收红）：
   - {'；'.join(_ai_etf_text(etf_anomalies))}
3. 用户科技防线：
   - {'；'.join(defense_lines)}
4. 全市场资金与估值交叉表：
   - 资金净流入Top5：{'；'.join(infl_lines) or '暂无数据'}
   - 资金净流出Top5：{'；'.join(outfl_lines) or '暂无数据'}
   - 极端高估警报（PE分位>80%）：{exp_txt}
   - 极端低估洼地（PE分位<20%）：{cheap_txt}
   - 低估且资金正在流入：{'；'.join(cheap_inflow)}

【账户约束】总资金{total_capital:,.0f}元，科技深套约{tech_position:,.0f}元（约{tech_position / total_capital * 100 if total_capital else 0:.0f}%），可用现金约{cash:,.0f}元。
【持仓明细】{holding_txt}

请严格按 Output Format 输出四个小节，全文不超过300字。"""


def build_local_conclusion(
    valuation_rows: list[dict[str, Any]],
    inflow: pd.DataFrame,
    outflow: pd.DataFrame,
    rotation: dict[str, Any],
    supports: list[dict[str, Any]],
    opportunities: list[dict[str, Any]],
    holdings: list[dict[str, Any]],
    resonance: dict[str, Any],
    events: list[dict[str, Any]],
    macro: dict[str, Any],
    msignal: dict[str, str],
    consec_inflows: list[dict[str, Any]],
    tech_position: float,
    cash: float,
    rebound_take_profit: float,
) -> str:
    elec = outflow[outflow["行业"].isin(TECH_FLOW_NAMES)]
    elec_text = (
        "、".join(f"{row['行业']}流出{_fmt_compact_yi(abs(float(row['净流入'])))}" for _, row in elec.iterrows())
        if not elec.empty
        else ""
    )

    # —— 宏观定性 ——
    if msignal["level"] == "red":
        qual = f"外围压制明显：美债10Y {macro['us10y']['value']:.2f}%（{macro['us10y']['chg']:+.2f}%）叠加美元走强，高估值科技股承压，严禁追高。"
    elif msignal["level"] == "yellow":
        qual = f"汇率端承压：离岸人民币{macro['usdcnh'].get('pips', 0):+.0f}点，外资流向不利核心资产。"
    elif msignal["level"] == "green":
        qual = "避险模式开启：黄金走强+美债收益率回落，防御资产（红利/黄金）价值凸显。"
    else:
        qual = "外围宏观中性，美元/美债/汇率均未出极端信号，主导权回到内部资金。"

    # —— 危险预警 ——
    weakest = min(supports, key=lambda item: float(item["dist"]))
    if weakest["dist"] <= 0:
        danger = f"{weakest['name']}已跌破20日支撑{weakest['low20']:.2f}，按纪律必须减半。"
    elif weakest["dist"] <= 3:
        danger = f"{weakest['name']}距20日支撑仅{weakest['dist']:.1f}%，收盘跌破{weakest['low20']:.2f}必须减半。"
    else:
        danger = f"{weakest['name']}距20日支撑{weakest['dist']:.1f}%，跌破{weakest['low20']:.2f}无条件减半。"
    if elec_text:
        danger += f"{elec_text}，深套科技切勿补仓。"
    hard_stops = [h for h in holdings if h["pnl_pct"] <= -8]
    if hard_stops:
        h = hard_stops[0]
        danger += f"{h['name']}持仓{h['pnl_pct']:+.1f}%触发-8%硬止损参考，建议减半。"
    if msignal["level"] == "red":
        danger += "美债飙升+科技流出共振，反弹即减仓。"
    if resonance["negatives"]:
        danger += f"双重利空：{resonance['negatives'][0]}临近。"

    # —— 高低切雷达 ——
    leads = [c for c in consec_inflows if c["pct"] is not None and c["pct"] < 40]
    if leads:
        clue = "、".join(
            f"{c['name']}（分位{c['pct']:.0f}%、3日流入{_fmt_compact_yi(c['total'])}）" for c in leads[:2]
        )
        clue += "——可纳入观察名单，等回踩不破再小仓试。"
    elif opportunities:
        c = opportunities[0]
        clue = f"{c['name']}（PE分位{c['pct']:.1f}%、3日流入{_fmt_yi(c['three_day'])}）——可纳入观察名单。"
    else:
        clue = "无（无板块同时满足低分位+连续流入，继续等右侧）。"
    crowded = [r for r in valuation_rows if r["pct"] >= 80]
    avoid = "、".join(f"{r['name']}(分位{r['pct']:.0f}%)" for r in crowded[:3]) or "无极端高估板块"

    # —— 今日执行纪律 ——
    reduce_amt = tech_position * 0.2
    below = [r["name"] for r in supports if not r["above_ma5"]]
    rules = [
        f"不买科技：{'、'.join(below) + '在5日线下方，' if below else ''}未见放量企稳，禁止补仓。",
        f"科技反弹超{rebound_take_profit:.0f}%先减约{reduce_amt:,.0f}元，跌破20日支撑无条件减半。",
        f"现金{cash:,.0f}元保持不动，不满仓、不加杠杆。",
    ]
    if msignal["level"] == "red":
        rules[0] = "外围风偏极差，今日科技一律不买，反弹只减不加。"
    upcoming = [e for e in events if _event_days(e) >= 0]
    if upcoming:
        nearest = min(upcoming, key=_event_days)
        rules.append(f"盯防{_event_days(nearest)}天后的{nearest['title']}，事件前不重仓。")

    return (
        f"### 【宏观定性】\n{qual}\n\n"
        f"### 【危险预警】\n{danger}\n\n"
        f"### 【高低切雷达】\n"
        f"- **避险区**：{avoid}——拥挤度过高，不建议追高。\n"
        f"- **潜在线索**：{clue}\n\n"
        f"### 【今日执行纪律】\n"
        + "\n".join(f"{idx}. {rule}" for idx, rule in enumerate(rules[:3], start=1))
    )


def _chat_completion(
    api_key: str,
    base_url: str,
    model: str,
    system: str,
    user: str,
    temperature: float = 0.15,
    max_tokens: int = 420,
) -> str:
    url = base_url.rstrip("/")
    if not url.endswith("/chat/completions"):
        url += "/chat/completions"
    response = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=45,
    )
    response.raise_for_status()
    payload = response.json()
    content = (payload["choices"][0]["message"]["content"] or "").strip()
    if not content:
        raise ValueError("模型返回空内容")
    return content


@st.cache_data(ttl=1800, show_spinner=False)
def call_llm(
    api_key: str,
    base_url: str,
    model: str,
    prompt: str,
    nonce: int,
) -> str:
    return _chat_completion(api_key, base_url, model, SYSTEM_PROMPT, prompt)


# ---------------------------------------------------------------------------
# 页面样式
# ---------------------------------------------------------------------------

st.markdown(
    """
<style>
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC",
                 "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
}
.stApp {
    background:
        radial-gradient(circle at 8% 0%, rgba(22, 163, 120, 0.07), transparent 26rem),
        radial-gradient(circle at 92% 8%, rgba(37, 99, 235, 0.06), transparent 28rem),
        #f4f6fb;
}
.block-container {
    max-width: 1520px;
    padding-top: 1.1rem;
    padding-bottom: 2rem;
}
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #ffffff;
    border: 1px solid #e4e9f2;
    border-radius: 18px;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.045);
}
.hero {
    display: flex;
    justify-content: space-between;
    gap: 16px;
    align-items: center;
    padding: 12px 18px;
    border-radius: 12px;
    color: #f8fafc;
    background:
        linear-gradient(120deg, rgba(7, 24, 48, 0.96), rgba(11, 58, 83, 0.94) 58%, rgba(9, 94, 85, 0.92)),
        #0b2747;
    overflow: hidden;
    position: relative;
}
.hero-left { position: relative; z-index: 1; }
.eyebrow {
    font-size: 0.56rem;
    letter-spacing: 0.14em;
    color: #8ddbd2;
    font-weight: 700;
    text-transform: uppercase;
}
.hero h1 {
    margin: 2px 0 0;
    font-size: 1.02rem;
    line-height: 1.2;
    letter-spacing: -0.01em;
    font-weight: 700;
}
.hero p { margin: 0; color: #bac8d9; font-size: 0.9rem; }
.hero-pills {
    position: relative;
    z-index: 1;
    display: flex;
    flex-wrap: wrap;
    justify-content: flex-end;
    gap: 8px;
    max-width: 430px;
}
.pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 5px 9px;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.09);
    border: 1px solid rgba(255, 255, 255, 0.14);
    color: #e2e8f0;
    font-size: 0.75rem;
    white-space: nowrap;
    backdrop-filter: blur(8px);
}
.pill b { color: #ffffff; font-weight: 700; }
.statusbar {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin: 12px 0 16px;
}
.status-chip {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    border: 1px solid #e1e7ef;
    background: rgba(255, 255, 255, 0.85);
    color: #526176;
    font-size: 0.7rem;
    padding: 6px 9px;
    border-radius: 999px;
}
.dot { width: 6px; height: 6px; border-radius: 50%; background: #16a36a; display: inline-block; }
.dot.warn { background: #d89016; }
.dot.demo { background: #7c8797; }
.macro-bar {
    border: 1px solid #e4e9f1;
    border-radius: 14px;
    background: #ffffff;
    padding: 8px 12px 7px;
    margin-bottom: 12px;
}
.macro-head {
    display: flex; justify-content: space-between; align-items: baseline;
    margin-bottom: 6px;
}
.macro-title { font-size: 0.72rem; font-weight: 700; color: #243247; letter-spacing: 0.02em; }
.macro-src { font-size: 0.55rem; color: #97a1b1; }
.macro-grid { display: grid; grid-template-columns: repeat(6, 1fr); gap: 6px; }
.macro-cell {
    border: 1px solid #edf1f6; border-radius: 8px; padding: 5px 7px;
    background: #fbfcfe;
}
.macro-cell .m-name { font-size: 0.56rem; color: #8a95a5; }
.macro-cell .m-value { font-size: 0.85rem; font-weight: 700; color: #1c2a3f; font-variant-numeric: tabular-nums; }
.macro-cell .m-chg { font-size: 0.56rem; font-weight: 600; }
.m-chg.up { color: #d64545; }
.m-chg.down { color: #0e9f6e; }
.macro-signal {
    margin-top: 7px; border-radius: 8px; padding: 6px 10px;
    font-size: 0.68rem; line-height: 1.45;
}
.macro-signal.red { background: #fff4f2; border: 1px solid #f4cbc5; color: #8e1f1f; }
.macro-signal.yellow { background: #fffaf0; border: 1px solid #f2dcb8; color: #8a5a12; }
.macro-signal.green { background: #f0faf4; border: 1px solid #bfe8d4; color: #14624a; }
.macro-signal.neutral { background: #f7f9fc; border: 1px solid #e6ecf4; color: #526176; }
.macro-signal b { display: block; font-size: 0.72rem; margin-bottom: 2px; }
@media (max-width: 900px) { .macro-grid { grid-template-columns: repeat(3, 1fr); } }
.section-kicker {
    color: #7c8797;
    font-size: 0.68rem;
    letter-spacing: 0.14em;
    font-weight: 700;
    text-transform: uppercase;
    margin-bottom: 2px;
}
.section-title {
    margin: 0 0 2px;
    color: #152238;
    font-size: 1.28rem;
    font-weight: 740;
    letter-spacing: -0.02em;
}
.section-sub {
    color: #7b8798;
    font-size: 0.76rem;
    margin: 0 0 12px;
}
.vhead {
    display: grid;
    grid-template-columns: 1.2fr 0.7fr 1fr 0.6fr;
    gap: 8px;
    color: #8a95a5;
    font-size: 0.65rem;
    letter-spacing: 0.04em;
    padding: 0 2px 7px;
    border-bottom: 1px solid #edf1f6;
}
.vrow { padding: 10px 2px 9px; border-bottom: 1px solid #f0f3f8; }
.vrow:last-child { border-bottom: none; }
.vtop { display: flex; justify-content: space-between; align-items: baseline; gap: 8px; }
.vname { color: #172033; font-size: 0.85rem; font-weight: 680; }
.vtag { color: #9aa4b3; font-size: 0.65rem; margin-left: 5px; font-weight: 500; }
.vright { display: flex; align-items: center; gap: 7px; }
.vpe { color: #263247; font-size: 0.82rem; font-weight: 700; font-variant-numeric: tabular-nums; }
.vpct { color: #8a95a5; font-size: 0.68rem; font-variant-numeric: tabular-nums; }
.vsub { color: #8a95a5; font-size: 0.66rem; margin: 2px 0 4px; font-variant-numeric: tabular-nums; }
.verdict { font-size: 0.68rem; font-weight: 650; margin: 3px 0 1px; }
.verdict.good { color: #0d7a56; }
.verdict.warn { color: #94600a; }
.verdict.bad { color: #b33232; }
.verdict.neutral { color: #7c8797; }
.support-sub { color: #97a1b1; font-size: 0.62rem; padding: 0 2px 6px 2px; margin-top: -3px; font-variant-numeric: tabular-nums; }
.breadth-line { font-size: 0.7rem; color: #526176; background: #f7f9fc; border: 1px solid #e6ecf4; border-radius: 8px; padding: 6px 10px; margin: 2px 0 8px; }
.breadth-line .b-up { color: #d64545; }
.breadth-line .b-down { color: #16a36a; }
.etf-prem { font-size: 0.66rem; color: #7c8797; margin-top: 6px; }
.etf-tag {
    display: inline-block;
    margin-left: 5px;
    padding: 0 4px;
    border-radius: 4px;
    font-size: 0.6rem;
    font-weight: 650;
    line-height: 1.6;
    cursor: help;
    border: 1px solid transparent;
}
.etf-tag.hot { color: #b33232; background: #fdeeec; border-color: #f5c9c4; }
.etf-tag.cool { color: #0d7a56; background: #eaf8f1; border-color: #bfe8d4; }
.etf-tag.flat { color: #6b7787; background: #f2f5f9; }
.etf-tag.na { color: #a7b0bd; background: transparent; }
.margin-head { display: flex; justify-content: space-between; align-items: baseline; margin: 2px 0 2px; }
.margin-value { font-size: 1.02rem; font-weight: 760; color: #172033; font-variant-numeric: tabular-nums; }
.margin-chg { font-size: 0.65rem; color: #7c8797; font-variant-numeric: tabular-nums; }
.margin-signal { font-size: 0.68rem; font-weight: 650; margin: 3px 0 4px; line-height: 1.5; }
.margin-signal.warn { color: #94600a; }
.margin-signal.good { color: #0d7a56; }
.margin-signal.neutral { color: #7c8797; }
.margin-alert {
    margin: 4px 0 2px; padding: 5px 9px; border-radius: 8px;
    font-size: 0.66rem; font-weight: 650; line-height: 1.55;
}
.margin-alert.red { background: #fdeceb; color: #a92c2c; border-left: 3px solid #e23b3b; }
.margin-alert.yellow { background: #fdf6e3; color: #8a5b06; border-left: 3px solid #e3a008; }
.margin-alert.green { background: #eaf7f0; color: #0d6b4c; border-left: 3px solid #16a374; }
.margin-alert.blue { background: #eaf1fb; color: #1f4f8f; border-left: 3px solid #2f6fb0; }
.margin-alert.gray { background: #f4f6f9; color: #66717f; border-left: 3px solid #b7c0cc; }
.data-stamp { color: #97a1b1; font-size: 0.64rem; margin: 6px 0 2px 2px; line-height: 1.5; }
.badge {
    display: inline-flex;
    align-items: center;
    border-radius: 999px;
    padding: 2px 7px;
    font-size: 0.63rem;
    font-weight: 700;
    line-height: 1.5;
    border: 1px solid transparent;
}
.badge.low { color: #087c55; background: #e8f8f0; border-color: #c6efdd; }
.badge.mid { color: #98630c; background: #fff7e5; border-color: #f6e2b5; }
.badge.high { color: #b33232; background: #fef0ef; border-color: #f8d4d0; }
.track {
    height: 5px;
    border-radius: 99px;
    background: #edf1f6;
    margin-top: 7px;
    position: relative;
    overflow: visible;
}
.fill { height: 100%; border-radius: 99px; opacity: 0.9; }
.fill.low { background: linear-gradient(90deg, #4fce9c, #0e9f6e); }
.fill.mid { background: linear-gradient(90deg, #f0c26d, #d89016); }
.fill.high { background: linear-gradient(90deg, #f08a82, #d64545); }
.gap-card {
    margin-top: 11px;
    padding: 12px 13px;
    border-radius: 14px;
    background: #f6f9fc;
    border: 1px solid #e7edf5;
}
.gap-title { color: #66748a; font-size: 0.68rem; letter-spacing: 0.06em; font-weight: 700; }
.gap-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 9px;
    margin: 9px 0 7px;
}
.gap-cell { background: #ffffff; border: 1px solid #edf1f7; border-radius: 11px; padding: 8px 9px; }
.gap-cell span { display: block; color: #8a95a5; font-size: 0.64rem; }
.gap-cell b { color: #172033; font-size: 0.92rem; font-variant-numeric: tabular-nums; }
.gap-note { color: #748196; font-size: 0.68rem; line-height: 1.55; }
.support-row {
    display: grid;
    grid-template-columns: 1.1fr 0.8fr 0.9fr 0.6fr;
    gap: 6px;
    align-items: center;
    padding: 8px 2px;
    border-bottom: 1px solid #f0f3f8;
    font-size: 0.69rem;
}
.support-row:last-child { border-bottom: none; }
.support-name { color: #25324a; font-weight: 650; }
.support-num { color: #5f6d82; font-variant-numeric: tabular-nums; }
.support-status { font-size: 0.62rem; font-weight: 700; }
.support-status.safe { color: #0e9f6e; }
.support-status.near { color: #d89016; }
.support-status.broken { color: #d64545; }
.flow-top {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 9px;
    margin-bottom: 8px;
}
.flow-card { border-radius: 13px; padding: 9px 11px; border: 1px solid #e8eef5; background: #fbfcfe; }
.flow-card.in { border-left: 3px solid #16a36a; }
.flow-card.out { border-left: 3px solid #d64545; }
.flow-label { color: #7f8a9b; font-size: 0.64rem; margin-bottom: 4px; }
.flow-value { color: #172033; font-size: 0.75rem; font-weight: 650; line-height: 1.65; }
.favg { color: #8a95a5; font-size: 0.66rem; font-weight: 500; }
.alert-card {
    border-radius: 13px;
    padding: 11px 13px;
    font-size: 0.74rem;
    line-height: 1.6;
    margin-top: 8px;
    border: 1px solid;
}
.alert-card.trigger { color: #8e1f1f; background: #fff4f2; border-color: #f4cbc5; }
.alert-card.warn { color: #8a5a12; background: #fffaf0; border-color: #f2dcb8; }
.alert-card.clear { color: #526176; background: #f7f9fc; border-color: #e6ecf4; }
.chip-row { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
.chip {
    border: 1px solid #e1e7ef; background: #fbfcfe; color: #3d4a5f;
    font-size: 0.68rem; border-radius: 999px; padding: 4px 10px;
}
.chip.hot { border-color: #f2b8c6; background: #fff5f7; color: #a02343; font-weight: 600; }
.alert-card b { display: block; margin-bottom: 3px; font-size: 0.72rem; letter-spacing: 0.02em; }
.event-card {
    display: grid;
    grid-template-columns: 50px 1fr;
    gap: 11px;
    padding: 10px 4px 10px 0;
    border-bottom: 1px solid #eef2f7;
    position: relative;
}
.event-card:last-child { border-bottom: none; }
.event-day {
    width: 44px;
    height: 44px;
    border-radius: 12px;
    background: #f2f6fb;
    border: 1px solid #e4ebf4;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    color: #1f3a55;
    font-weight: 750;
    font-variant-numeric: tabular-nums;
}
.event-day small { font-size: 0.52rem; color: #8a95a5; font-weight: 500; margin-top: 1px; }
.event-card.urgent .event-day { background: #fff1ef; border-color: #f3c9c4; color: #bb3131; }
.event-card.warn .event-day { background: #fff8e8; border-color: #f2ddb0; color: #9a660b; }
.event-body { min-width: 0; }
.event-meta { color: #8a95a5; font-size: 0.63rem; margin-bottom: 2px; }
.event-title { color: #1d293d; font-size: 0.81rem; font-weight: 680; margin-bottom: 2px; }
.event-impact { color: #758296; font-size: 0.66rem; line-height: 1.45; }
.event-badge { font-size: 0.61rem; color: #7f8a9b; margin-top: 4px; display: inline-block; }
.event-card.urgent .event-badge { color: #bb3131; }
.event-card.warn .event-badge { color: #9a660b; }
.ai-panel {
    border-radius: 14px;
    background: linear-gradient(180deg, #0d2038, #102a3d);
    color: #e9f0f7;
    padding: 14px 15px;
    border: 1px solid rgba(141, 219, 210, 0.17);
}
.ai-panel .ai-meta { color: #7fb8b2; font-size: 0.63rem; letter-spacing: 0.08em; margin-bottom: 7px; }
.ai-panel .ai-body { font-size: 0.78rem; line-height: 1.7; }
.ai-panel .ai-body strong { color: #8de0d1; }
.ai-local {
    display: inline-block;
    margin-top: 7px;
    color: #a9b6c7;
    font-size: 0.62rem;
    border-top: 1px solid rgba(255,255,255,0.08);
    padding-top: 6px;
}
.footer {
    color: #8b96a6;
    font-size: 0.67rem;
    line-height: 1.7;
    border-top: 1px solid #e7ecf4;
    margin-top: 18px;
    padding-top: 12px;
}
.side-brand { color: #64748b; font-size: 0.68rem; letter-spacing: 0.14em; font-weight: 700; margin-bottom: 2px; }
.side-title { color: #14213a; font-size: 1.05rem; font-weight: 760; margin-bottom: 10px; }
.side-note { color: #7c8798; font-size: 0.68rem; line-height: 1.55; }
@media (max-width: 900px) {
    .hero { flex-direction: column; align-items: flex-start; padding: 22px; }
    .hero-pills { justify-content: flex-start; max-width: none; }
    .vhead { grid-template-columns: 1.2fr 0.65fr 1fr 0.55fr; }
}
/* ---- 手机窄屏适配（<=640px / <=430px）---- */
@media (max-width: 640px) {
    .block-container {
        padding-left: 0.78rem; padding-right: 0.78rem;
        padding-top: 0.6rem; padding-bottom: 1.4rem;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 14px; }
    .hero { padding: 14px 14px; gap: 8px; }
    .hero h1 { font-size: 0.98rem; }
    .macro-bar { padding: 6px 7px; border-radius: 12px; }
    .macro-grid { grid-template-columns: repeat(3, 1fr); gap: 4px; }
    .macro-cell { padding: 3px 5px; border-radius: 6px; }
    .macro-cell .m-name { font-size: 0.5rem; }
    .macro-cell .m-value { font-size: 0.74rem; }
    .macro-cell .m-chg { font-size: 0.5rem; }
    .macro-signal { font-size: 0.62rem; padding: 5px 8px; margin-top: 5px; }
    .macro-head { margin-bottom: 4px; }
    .section-title { font-size: 1.1rem; }
    .section-sub { font-size: 0.71rem; }
    .vtop { flex-wrap: wrap; row-gap: 2px; }
    .vright { gap: 5px; }
    .vhead { font-size: 0.58rem; gap: 5px; }
    .support-row { grid-template-columns: 1.2fr 0.9fr 1fr 0.7fr; font-size: 0.64rem; }
    .gap-grid, .flow-top { grid-template-columns: 1fr; }
    .event-card { grid-template-columns: 42px 1fr; gap: 8px; }
    .ai-panel { padding: 12px; }
    .statusbar { gap: 5px; margin: 8px 0 12px; }
}
@media (max-width: 430px) {
    .vhead { display: none; }
    .vtag { display: none; }
    .etf-tag { font-size: 0.58rem; margin-left: 4px; }
    .macro-cell .m-name { font-size: 0.52rem; }
    .macro-cell .m-value { font-size: 0.76rem; }
    .status-chip { font-size: 0.63rem; padding: 5px 8px; }
    .hero-pills { gap: 5px; }
    .pill { font-size: 0.68rem; padding: 4px 7px; }
}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# 侧边栏
# ---------------------------------------------------------------------------

st.sidebar.markdown(
    """
<div class="side-brand">PERSONAL QUANT RADAR / 01</div>
<div class="side-title">控制台</div>
<div class="side-note">先保命，再谈收益。系统只给条件化的防守与机会信号。</div>
""",
    unsafe_allow_html=True,
)

mode_label = st.sidebar.radio(
    "数据模式",
    ["自动（真实数据优先）", "演示数据"],
    index=0,
    help="自动模式会先请求实时数据；接口不可用时自动切到内置演示数据。",
)
mode = "auto" if mode_label.startswith("自动") else "demo"

signal_mode_label = st.sidebar.radio(
    "估值信号灯口径",
    ["历史分位（推荐）", "绝对PE"],
    index=0,
    help="分位=跟自己近5年历史比（<20%绿/>80%红）；绝对=按PE绝对值（<15绿/>40红）。银行红利类在分位口径下会显示高估，因为近两年涨幅大。",
)
signal_mode = "分位" if signal_mode_label.startswith("历史") else "绝对"

st.sidebar.checkbox(
    "顶部显示资金概览",
    value=False,
    key="show_header",
    help="默认关闭（低调模式）。开启后页头显示总资金/科技仓位/可用子弹。",
)

st.sidebar.divider()
st.sidebar.markdown("**账户约束**")
total_capital = st.sidebar.number_input(
    "总可投资金（元）", min_value=0.0, value=100_000.0, step=10_000.0, format="%.0f"
)
tech_position = st.sidebar.number_input(
    "科技深套市值（元）", min_value=0.0, value=40_000.0, step=5_000.0, format="%.0f"
)
cash = max(total_capital - tech_position, 0.0)
st.sidebar.caption(f"可用子弹约 ¥{cash:,.0f} · 科技仓位占比 {tech_position / total_capital * 100 if total_capital else 0:.1f}%")

rebound_take_profit = st.sidebar.slider(
    "反弹做 T 阈值（%）", min_value=1, max_value=6, value=3, step=1,
    help="深套科技仓位反弹达到该幅度，按纪律减 1/5。",
)

with st.sidebar.expander("我的持仓（选填 · 盈亏监控）", expanded=False):
    st.caption("成本按对应指数点位填写，系统按现价估算盈亏")
    holding_inputs: list[dict[str, Any]] = []
    for i in range(3):
        h_name = st.selectbox(
            f"持仓 {i+1}", ["—", "科创50", "科创芯片", "通信"], key=f"h_name_{i}"
        )
        hc1, hc2 = st.columns(2)
        with hc1:
            h_cost = st.number_input(
                f"成本点位 {i+1}", min_value=0.0, value=0.0, step=10.0,
                format="%.2f", key=f"h_cost_{i}",
            )
        with hc2:
            h_amt = st.number_input(
                f"市值(元) {i+1}", min_value=0.0, value=0.0, step=1000.0,
                format="%.0f", key=f"h_amt_{i}",
            )
        if h_name != "—" and h_cost > 0 and h_amt > 0:
            holding_inputs.append({"name": h_name, "cost": h_cost, "amount": h_amt})

st.sidebar.divider()
_saved_cfg = load_config()
_saved_key = str(_saved_cfg.get("api_key", ""))
_secret_key = get_secret("DEEPSEEK_API_KEY")
with st.sidebar.expander("AI 接口（DeepSeek / OpenAI 兼容）", expanded=False):
    typed_key = st.text_input(
        "API Key",
        value=os.getenv("DEEPSEEK_API_KEY", "") or _saved_key,
        type="password",
        help="云端部署请把 Key 放在 st.secrets（DEEPSEEK_API_KEY），这里留空即可；本地可临时填写。",
    ).strip()
    model = st.text_input(
        "模型",
        value=os.getenv("DEEPSEEK_MODEL", "")
        or get_secret("DEEPSEEK_MODEL", "")
        or str(_saved_cfg.get("model", "deepseek-chat")),
    ).strip()
    base_url = st.text_input(
        "Base URL",
        value=os.getenv("DEEPSEEK_BASE_URL", "")
        or get_secret("DEEPSEEK_BASE_URL", "")
        or str(_saved_cfg.get("base_url", "https://api.deepseek.com")),
        help="填到 /chat/completions 之前，例如 https://api.deepseek.com 或 https://api.deepseek.com/v1。",
    ).strip()
    if _secret_key and not typed_key:
        st.caption("已从 st.secrets 读取云端 Key（不显示明文），留空即使用。")
    remember_key = st.checkbox(
        "记住 API Key（仅本机，存 data/config.json，不会提交到 GitHub）",
        value=bool(_saved_key),
    )
    if remember_key and typed_key:
        if (
            typed_key != _saved_key
            or model != _saved_cfg.get("model", "deepseek-chat")
            or base_url != _saved_cfg.get("base_url", "https://api.deepseek.com")
        ):
            save_config({"api_key": typed_key, "model": model, "base_url": base_url})
            st.caption("已记住，下次启动自动填充。")
        elif _saved_key:
            st.caption(f"已记住本机 Key（···{_saved_key[-4:]}）")
    elif not remember_key and _saved_key:
        delete_config()
        st.caption("已清除本机保存的 Key。")
    # 优先级：手动填写 > 本机记住 > st.secrets 云端 Key
    api_key = typed_key or _saved_key or _secret_key

if "refresh_token" not in st.session_state:
    st.session_state.refresh_token = 0
if "ai_nonce" not in st.session_state:
    st.session_state.ai_nonce = 0
if "data_stamp" not in st.session_state:
    st.session_state.data_stamp = None

if st.sidebar.button("刷新盘中数据", use_container_width=True):
    # 只清盘中高频缓存（资金流/涨跌家数/ETF溢价/宏观），
    # 日更数据（估值/科技防线/两融/ETF日线）当天已有缓存时继续复用。
    for _cached_fn in (load_flow, load_breadth, load_etf_premium, load_macro):
        try:
            _cached_fn.clear()
        except Exception:
            pass
    st.session_state.data_stamp = None
    st.rerun()

if st.sidebar.button("全量刷新（含估值/防线）", use_container_width=True):
    # 手动强制重拉日更数据；仅在你怀疑估值/两融数据卡住时使用
    st.session_state.refresh_token += 1
    st.session_state.ai_nonce += 1
    st.session_state.data_stamp = None
    st.cache_data.clear()
    st.rerun()

st.sidebar.divider()
st.sidebar.markdown(
    """
<div class="side-note">
<b>数据源</b><br>
估值：韭圈儿公开接口 · 近 5 年 PE 分位<br>
资金流：AkShare / 东方财富行业资金流<br>
事件：代码内置或 events.json<br><br>
<b>纪律</b><br>
不追高 · 深套不补仓 · 分批网格 · 破位减半
</div>
""",
    unsafe_allow_html=True,
)
st.sidebar.caption("仅供个人研究，不构成投资建议。")


# ---------------------------------------------------------------------------
# 数据装配
# ---------------------------------------------------------------------------

with st.spinner("读取宏观指标…"):
    macro, macro_meta = load_macro(st.session_state.refresh_token, mode)
msignal = macro_signal(macro)


# ---------------------------------------------------------------------------
# 页头（默认低调，不显示资金数字；侧边栏可开启）
# ---------------------------------------------------------------------------

_hero_pills = ""
if st.session_state.get("show_header"):
    _hero_pills = f"""
    <span class="pill">总资金 <b>¥{total_capital:,.0f}</b></span>
    <span class="pill">科技深套 <b>¥{tech_position:,.0f}</b></span>
    <span class="pill">可用子弹 <b>¥{cash:,.0f}</b></span>
    <span class="pill">纪律 <b>破位即减</b></span>"""
st.markdown(
    f"""
<div class="hero">
  <div class="hero-left">
    <div class="eyebrow">PERSONAL QUANT ROTATION · RISK RADAR</div>
    <h1>个人量化避坑雷达</h1>
  </div>
  <div class="hero-pills">{_hero_pills}</div>
</div>
""",
    unsafe_allow_html=True,
)

# 数据最后更新时间占位：等重型数据加载完再填充，位置保持在标题下方
_data_stamp_slot = st.empty()


# ---------------------------------------------------------------------------
# 宏观联动看板（最先渲染，首屏秒出）
# ---------------------------------------------------------------------------

def _macro_cell(key: str, unit: str = "") -> str:
    m = macro[key]
    chg = float(m.get("chg") or 0)
    cls = "up" if chg > 0 else "down"
    arrow = "▲" if chg > 0 else "▼"
    extra = ""
    value_txt = f"{float(m['value']):,.2f}{unit}"
    if key == "usdcnh":
        pips = float(m.get("pips", 0))
        extra = f"（{pips:+.0f}点 · {'人民币贬值' if pips >= 0 else '人民币升值'}）"
    elif key == "tips":
        extra = f"（{chg:+.0f}bp）"
        chg_txt = f"{arrow} {chg:+.1f}bp"
        return f"""
<div class="macro-cell">
  <div class="m-name">{m['name']}</div>
  <div class="m-value">{value_txt}</div>
  <div class="m-chg {cls}">{chg_txt}</div>
</div>"""
    elif key == "south":
        v = float(m["value"])
        scls = "up" if v > 0 else "down"
        return f"""
<div class="macro-cell">
  <div class="m-name">{m['name']}</div>
  <div class="m-value">{v:+,.0f}亿</div>
  <div class="m-chg {scls}">{m.get('asof', '')}</div>
</div>"""
    return f"""
<div class="macro-cell">
  <div class="m-name">{m['name']}</div>
  <div class="m-value">{value_txt}</div>
  <div class="m-chg {cls}">{arrow} {chg:+.2f}%{extra}</div>
</div>"""


st.markdown(
    f"""
<div class="macro-bar">
  <div class="macro-head">
    <span class="macro-title">跨市场宏观联动</span>
    <span class="macro-src">{macro_meta['source']} · {macro_meta['note']} · 新浪/中债/FRED/东财</span>
  </div>
  <div class="macro-grid">
    {_macro_cell('dxy')}
    {_macro_cell('us10y', '%')}
    {_macro_cell('tips', '%')}
    {_macro_cell('usdcnh')}
    {_macro_cell('gold')}
    {_macro_cell('south')}
  </div>
  <div class="macro-signal {msignal['level']}"><b>{'🔴' if msignal['level']=='red' else ('🟡' if msignal['level']=='yellow' else ('🟢' if msignal['level']=='green' else '⚪'))} {msignal['title']}</b>{msignal['text']}</div>
</div>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# 重型数据加载（宏观之后的估值/资金/防线）
# ---------------------------------------------------------------------------

with st.spinner("正在读取估值、资金流与科技防线…（首次约10~20秒）"):
    valuation_rows, valuation_meta = load_valuation(
        st.session_state.refresh_token, mode
    )
    flow_today, flow_today_meta = load_flow(
        "今日", st.session_state.refresh_token, mode
    )
    supports, support_meta = load_support(
        st.session_state.refresh_token, mode
    )
    if api_key and _events_stale(7) and not st.session_state.get("events_auto_done"):
        # 每周（文件超过 7 天）自动让 DeepSeek 重建一次事件日历，无需人工录入
        st.session_state.events_auto_done = True
        _, _event_err = generate_events(api_key, base_url, model)
        st.session_state.events_auto_error = _event_err
    events, event_source = load_events()
    # 溢价率与两融互不依赖，并行取数，避免串行拖慢首屏
    with ThreadPoolExecutor(max_workers=2) as pool:
        premium_future = pool.submit(
            load_etf_premium, st.session_state.refresh_token, mode
        )
        margin_future = pool.submit(
            load_margin, st.session_state.refresh_token, mode
        )
        etf_premiums = premium_future.result()
        margin_points, margin_meta = margin_future.result()

    # 三只宽基ETF日线并行预取（按标的分开缓存，切换时不重复请求）
    etf_anomalies: dict[str, tuple[pd.DataFrame, dict[str, Any]]] = {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            item["code"]: pool.submit(
                load_etf_anomaly, item["code"], st.session_state.refresh_token, mode
            )
            for item in BROAD_ETF_UNIVERSE
        }
        for code, future in futures.items():
            try:
                etf_anomalies[code] = future.result()
            except Exception as exc:
                etf_anomalies[code] = (
                    pd.DataFrame(),
                    {"source": "ETF日线暂不可用", "is_demo": False,
                     "asof": "", "errors": [_clip_error(exc)]},
                )

sw_today = _sw_frame(flow_today)
inflow_today, outflow_today = _flow_extremes(sw_today)
history, history_meta = get_flow_history(flow_today, flow_today_meta)
rotation = detect_rotation(history)
daily_rotation = detect_daily_rotation(sw_today)
consec_inflows = find_consecutive_inflows(history, valuation_rows)
opportunities, opportunity_note = find_opportunities(history, valuation_rows)
resonance = find_resonance(events, opportunities, sw_today)

# 数据最后更新时间（刷新后重置；其余切换操作读缓存，不会刷新这个时间）
if st.session_state.get("data_stamp") is None:
    st.session_state.data_stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
_stamp_bits = [f"数据最后更新时间：{st.session_state.data_stamp}"]
if valuation_meta.get("asof"):
    _stamp_bits.append(f"估值 {valuation_meta['asof']}")
if flow_today_meta.get("asof"):
    _stamp_bits.append(f"资金流 {flow_today_meta['asof']}")
if any((valuation_meta.get("is_demo"), flow_today_meta.get("is_demo"))):
    _stamp_bits.append("含演示数据")
_data_stamp_slot.markdown(
    '<div class="data-stamp">' + "　·　".join(_stamp_bits)
    + "　·　盘中数据缓存 5 分钟，点侧边栏「刷新盘中数据」可立即更新</div>",
    unsafe_allow_html=True,
)

# 持仓盈亏：用支撑模块的现价估算
_price_map = {row["name"]: float(row["latest"]) for row in supports}
holdings: list[dict[str, Any]] = []
for h in holding_inputs:
    if h["name"] not in _price_map:
        continue
    latest = _price_map[h["name"]]
    pnl_pct = (latest / h["cost"] - 1) * 100
    holdings.append({**h, "latest": latest, "pnl_pct": pnl_pct})

# 科技持仓 × 资金抽血 联动警告
_tech_drain = sw_today[
    (sw_today["行业"].isin(TECH_FLOW_NAMES)) & (sw_today["净流入"] < 0)
]
tech_drain_text = "、".join(
    f"{row['行业']}净流出{_fmt_compact_yi(abs(float(row['净流入'])))}"
    for _, row in _tech_drain.iterrows()
)


# ---------------------------------------------------------------------------
# 数据状态条
# ---------------------------------------------------------------------------

valuation_dot = "demo" if valuation_meta["is_demo"] else "live"
flow_dot = "demo" if flow_today_meta["is_demo"] else "live"
support_dot = "demo" if support_meta["is_demo"] else "live"
_lag = valuation_meta.get("lag_days", 0)
_stale_cls = "dot demo" if valuation_meta.get("stale") else valuation_dot
_stale_txt = f" · 落后{_lag}天（源端未刷新）" if valuation_meta.get("stale") else ""
st.markdown(
    f"""
<div class="statusbar">
  <span class="status-chip"><span class="dot {_stale_cls}"></span>估值：{valuation_meta['source']} · {valuation_meta['asof']}{_stale_txt}</span>
  <span class="status-chip"><span class="dot {flow_dot}"></span>资金流：{flow_today_meta['source']} · {flow_today_meta['asof']}</span>
  <span class="status-chip"><span class="dot {support_dot}"></span>防线：{support_meta['source']} · {support_meta['asof']}</span>
  <span class="status-chip"><span class="dot"></span>事件：{event_source}</span>
</div>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# 三栏布局
# ---------------------------------------------------------------------------

left_col, mid_col, right_col = st.columns([1.1, 1.2, 1.0], gap="large")


# ----------------------------- 01 估值水位 -------------------------------
with left_col:
    with st.container(border=True):
        st.markdown(
            """
<div class="section-kicker">01 / VALUATION</div>
<div class="section-title">核心持仓估值水位</div>
<div class="section-sub">PE 近5年分位定高低：&lt;20% 低估(绿) · &gt;80% 高估(红) · 韭圈儿整体法口径</div>
""",
            unsafe_allow_html=True,
        )

        core_rows = [row for row in valuation_rows if row["name"] in CORE_NAMES]
        valuation_map = {row["name"]: row for row in valuation_rows}
        tech_rows = [valuation_map[name] for name in TECH_VALUATION_NAMES if name in valuation_map]
        low_rows = [valuation_map[name] for name in LOW_VALUATION_NAMES if name in valuation_map]
        tech_pct = sum(row["pct"] for row in tech_rows) / len(tech_rows) if tech_rows else 0
        low_pct = sum(row["pct"] for row in low_rows) / len(low_rows) if low_rows else 0
        tech_pe = sum(row["pe"] for row in tech_rows) / len(tech_rows) if tech_rows else 0
        low_pe = sum(row["pe"] for row in low_rows) / len(low_rows) if low_rows else 0

        st.markdown(
            '<div class="vhead"><span>指数</span><span>PE-TTM</span><span>PE 5年分位</span><span>信号</span></div>',
            unsafe_allow_html=True,
        )

        rows_html = []
        for row in core_rows:
            signal = _valuation_signal(float(row["pct"]), signal_mode, row.get("pe"))
            pct = max(0.0, min(float(row["pct"]), 100.0))
            pb_txt = f"{row['pb']:.2f}" if row.get("pb") is not None else "—"
            pbp_txt = f"{row['pb_pct']:.0f}%" if row.get("pb_pct") is not None else "—"
            dy_txt = f"{row['dy']:.2f}%" if row.get("dy") is not None else "—"
            verdict_txt, verdict_cls = _valuation_verdict(row)
            prem_html = _premium_tag(etf_premiums.get(VALUATION_ETF.get(row["name"], "")))
            rows_html.append(
                f"""
<div class="vrow">
  <div class="vtop">
    <div><span class="vname">{row['name']}</span>{prem_html}<span class="vtag">{row['tag']}</span></div>
    <div class="vright">
      <span class="vpe">{row['pe']:.1f}x</span>
      <span class="vpct">PE分位 {pct:.0f}%</span>
      <span class="badge {signal['cls']}">{signal['label']}</span>
    </div>
  </div>
  <div class="vsub">PB {pb_txt} · PB分位 {pbp_txt} · 股息率 {dy_txt}</div>
  <div class="verdict {verdict_cls}">→ {verdict_txt}</div>
  <div class="track"><div class="fill {signal['cls']}" style="width:{pct:.1f}%"></div></div>
</div>
"""
            )
        st.markdown("".join(rows_html), unsafe_allow_html=True)
        st.caption(
            "名称旁「溢价/折价」标签 = 对应场内 ETF 实时溢价率（东财 IOPV）；"
            "溢价 >1.5% 红色勿追、改买场外，折价 <-1% 绿色注意机会，取不到显示「溢价 —」。"
        )

        gap_pp = tech_pct - low_pct
        gap_pe = tech_pe - low_pe
        if gap_pp >= 20:
            gap_note = "结论：科技组估值分位显著高于低位组，回调风险未释放完毕，低位组性价比凸显。"
        elif gap_pp <= -20:
            gap_note = "结论：科技组估值分位已低于低位组，属“便宜但深套”，仍需等资金右侧信号再补仓。"
        else:
            gap_note = "结论：两组估值分位接近，性价比差距不大，轮动以资金流信号为准。"
        st.markdown(
            f"""
<div class="gap-card">
  <div class="gap-title">估值剪刀差 · 深套科技 vs 低位候选</div>
  <div class="gap-grid">
    <div class="gap-cell"><span>科技组 PE / 分位</span><b>{tech_pe:.1f}x · {tech_pct:.1f}%</b></div>
    <div class="gap-cell"><span>低位组 PE / 分位</span><b>{low_pe:.1f}x · {low_pct:.1f}%</b></div>
  </div>
  <div class="gap-note">PE 差 <b>{gap_pe:+.1f}x</b> · 分位差 <b>{gap_pp:+.1f} pct</b><br><b style="color:#173f35;">{gap_note}</b></div>
</div>
""",
            unsafe_allow_html=True,
        )

    with st.expander("02 / UNIVERSE · 全市场扩展监控池（9 个板块 · 点击展开）", expanded=False):
        st.markdown(
            """
<div class="section-sub">金融/消费/周期/制造 · ETF 标的指数 · 近5年分位定高低</div>
""",
            unsafe_allow_html=True,
        )
        ext_rows = [r for r in valuation_rows if r["name"] in EXTENDED_NAMES]
        st.markdown(
            '<div class="vhead"><span>板块（ETF标的）</span><span>PE</span><span>PE 5年分位</span><span>信号</span></div>',
            unsafe_allow_html=True,
        )
        ext_html = []
        for row in ext_rows:
            signal = _valuation_signal(float(row["pct"]), signal_mode, row.get("pe"))
            pct = max(0.0, min(float(row["pct"]), 100.0))
            pb_txt = f"{row['pb']:.2f}" if row.get("pb") is not None else "—"
            pbp_txt = f"{row['pb_pct']:.0f}%" if row.get("pb_pct") is not None else "—"
            dy_txt = f"{row['dy']:.2f}%" if row.get("dy") is not None else "—"
            verdict_txt, verdict_cls = _valuation_verdict(row)
            prem_html = _premium_tag(etf_premiums.get(VALUATION_ETF.get(row["name"], "")))
            ext_html.append(
                f"""
<div class="vrow">
  <div class="vtop">
    <div><span class="vname">{row['name']}</span>{prem_html}<span class="vtag">{row['tag']}</span></div>
    <div class="vright">
      <span class="vpe">{row['pe']:.1f}x</span>
      <span class="vpct">PE分位 {pct:.0f}%</span>
      <span class="badge {signal['cls']}">{signal['label']}</span>
    </div>
  </div>
  <div class="vsub">PB {pb_txt} · PB分位 {pbp_txt} · 股息率 {dy_txt}</div>
  <div class="verdict {verdict_cls}">→ {verdict_txt}</div>
  <div class="track"><div class="fill {signal['cls']}" style="width:{pct:.1f}%"></div></div>
</div>
"""
            )
        st.markdown("".join(ext_html), unsafe_allow_html=True)
        st.caption("监控池可在程序目录新建 watchlist.json 自行增删（格式同代码内 VALUATION_UNIVERSE）")

with mid_col:
    with st.container(border=True):
        st.markdown(
            """
<div class="section-kicker">02 / FUND FLOW</div>
<div class="section-title">主力资金异动与高低切</div>
<div class="section-sub">申万一级映射 · 只展示流入前 5 与流出前 5</div>
""",
            unsafe_allow_html=True,
        )
        breadth = load_breadth(st.session_state.refresh_token, mode)
        st.markdown(
            f"""
<div class="breadth-line">涨停 <b class="b-up">{breadth['limit_up']}</b> 家 · 跌停 <b class="b-down">{breadth['limit_down']}</b> 家 · 上涨/下跌 {breadth['up']}/{breadth['down']} · 中位涨幅 {breadth['median']:+.2f}%（{breadth['source']}）</div>
""",
            unsafe_allow_html=True,
        )
        period = st.radio(
            "资金观察窗口",
            ["今日", "5日"],
            horizontal=True,
            label_visibility="collapsed",
        )
        if period == "5日":
            flow_view, flow_view_meta = load_flow(
                "5日", st.session_state.refresh_token, mode
            )
        else:
            flow_view, flow_view_meta = flow_today, flow_today_meta

        sw_view = _sw_frame(flow_view)
        unavailable = bool(flow_view_meta.get("unavailable"))
        if unavailable:
            inflow = pd.DataFrame(columns=["行业", "净流入", "涨跌幅"])
            outflow = inflow.copy()
        else:
            inflow, outflow = _flow_extremes(sw_view)

        if flow_view_meta.get("is_demo"):
            err = (flow_view_meta.get("errors") or [""])[0]
            st.error(
                "⚠ 当前资金流为**演示数据**（数值固定，非真实行情），请点右上角刷新重试。"
                + (f"　接口状态：{_friendly_error(err)}" if err else "")
            )
        elif unavailable:
            err = (flow_view_meta.get("errors") or [""])[0]
            st.warning(
                "5日资金流暂不可用："
                + _friendly_error(err)
                + "。为避免展示假数据，本页暂不显示 5 日柱状图，请切到「今日」或稍后点右上角刷新。"
            )
        elif flow_view_meta.get("stale") and period == "5日":
            st.caption(
                f"实时接口限流，当前显示本地最近一次真实 5 日数据（{flow_view_meta['asof']}），未使用任何假数据。"
            )

        avg5_map: dict[str, float] | None = None
        avg5_days = 5
        avg5_note = ""
        if period == "今日":
            flow_5d, meta_5d = load_flow("5日", st.session_state.refresh_token, mode)
            if meta_5d.get("unavailable") or meta_5d.get("is_demo"):
                avg5_note = (
                    "当前为演示模式，不计算 5 日均值。"
                    if mode == "demo"
                    else "5日均值暂缺：东方财富接口限流（约 5 分钟恢复）。"
                    "今日数据不受影响，稍后点右上角刷新即可恢复。"
                )
            else:
                avg5_days = int(meta_5d.get("days") or 5)
                avg5_map = {
                    str(row["行业"]): float(row["净流入"]) / avg5_days
                    for _, row in _sw_frame(flow_5d).iterrows()
                }
                if meta_5d.get("stale"):
                    avg5_note = (
                        f"5日均值取自本地最近一次真实快照（{meta_5d['asof']}），"
                        "实时接口限流中，数值不是假的。"
                    )

        if not unavailable:
            days_5d = int(flow_view_meta.get("days") or 5)
            period_label = (
                "单日"
                if period == "今日"
                else ("近5日累计" if days_5d >= 5 else f"近{days_5d}日累计")
            )
            avg_label = "5日均" if avg5_days >= 5 else f"近{avg5_days}日均"

            st.markdown(
                f"""
<div class="flow-top">
  <div class="flow-card in"><div class="flow-label">主力净流入 TOP5</div><div class="flow-value">{_flow_card_lines(inflow, avg5_map, avg_label)}</div></div>
  <div class="flow-card out"><div class="flow-label">主力净流出 TOP5</div><div class="flow-value">{_flow_card_lines(outflow, avg5_map, avg_label)}</div></div>
</div>
""",
                unsafe_allow_html=True,
            )

            chart_frame = pd.concat([inflow, outflow], ignore_index=True)
            chart_frame = chart_frame.drop_duplicates(subset=["行业"]).sort_values("净流入")
            if go is not None and not chart_frame.empty:
                values = chart_frame["净流入"] / 1e8
                labels = chart_frame["行业"].tolist()
                colors = ["#16a36a" if value >= 0 else "#d64545" for value in values]

                fig = go.Figure(
                    go.Bar(
                        x=values,
                        y=labels,
                        orientation="h",
                        marker_color=colors,
                        cliponaxis=False,
                        hovertemplate="%{y}<br>主力净流入：%{x:.2f} 亿元<extra></extra>",
                    )
                )
                # 数值统一放在图表右侧外部标注，避免挤压柱内字号
                annotations = []
                for name, value in zip(labels, values):
                    if avg5_map and name in avg5_map:
                        txt = f"{value:+.2f}亿 · 均{avg5_map[name] / 1e8:+.2f}亿"
                    else:
                        txt = f"{value:+.2f}亿"
                    annotations.append(
                        dict(
                            x=1.01,
                            xref="paper",
                            xanchor="left",
                            y=name,
                            yref="y",
                            yanchor="middle",
                            text=txt,
                            showarrow=False,
                            font=dict(size=12, color="#0d7a56" if value >= 0 else "#b33434"),
                        )
                    )
                fig.update_layout(
                    title=dict(
                        text=(
                            ("⚠ 演示数据（非真实）｜" if flow_view_meta.get("is_demo") else "")
                            + f"{period_label}主力净流入（亿元）"
                            f"{f' · 均=近{avg5_days}日日均' if avg5_map else ''}"
                            f" · {flow_view_meta['source']} {flow_view_meta['asof']}"
                        ),
                        font=dict(size=11.5, color="#b33434" if flow_view_meta.get("is_demo") else "#526176"),
                        x=0,
                    ),
                    annotations=annotations,
                    height=342,
                    margin=dict(l=8, r=124, t=30, b=28),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="PingFang SC, Microsoft YaHei, sans-serif", size=11, color="#4c5b70"),
                    xaxis_title=None,
                    xaxis=dict(
                        zeroline=True,
                        zerolinecolor="#dbe2eb",
                        gridcolor="#edf1f6",
                        tickfont=dict(size=10, color="#7b8798"),
                        title_font=dict(size=10, color="#7b8798"),
                    ),
                    yaxis=dict(
                        autorange="reversed",
                        tickfont=dict(size=10.5, color="#25324a"),
                        showgrid=False,
                    ),
                    bargap=0.32,
                )
                _render_plotly(fig, config={"displayModeBar": False}, key="flow_chart")
            else:
                st.caption(f"{period_label}主力净流入（亿元） · 数据日期 {flow_view_meta['asof']}")
                st.bar_chart(chart_frame.set_index("行业")["净流入"] / 1e8)

            if avg5_note:
                st.caption(avg5_note)

        if etf_premiums:
            prem_txt = " · ".join(
                f"{name} {etf_premiums[code]['premium']:+.2f}%"
                for code, name in FLOW_ETFS.items()
                if code in etf_premiums
            )
            if prem_txt:
                st.markdown(
                    f"<div class='etf-prem'>ETF 实时溢价率（东财IOPV）：{prem_txt} ｜ 溢价 &gt;1.5% 勿追，折价 &lt;-1% 注意机会</div>",
                    unsafe_allow_html=True,
                )

        alert_cls = "trigger" if rotation["triggered"] else "clear"
        alert_title = "高低切预警已触发" if rotation["triggered"] else "高低切预警未触发"
        st.markdown(
            f"""
<div class="alert-card {alert_cls}">
  <b>{alert_title}</b>
  {rotation['message']}<br>
  <span style="opacity:.72;font-size:.65rem;">
  观察样本：{history_meta['source']} · 电子/通信需连续 3 日净流出；低位板块需连续 3 日流入前三。
  </span>
</div>
""",
            unsafe_allow_html=True,
        )

        if daily_rotation["triggered"] and not rotation["triggered"]:
            st.markdown(
                f"""
<div class="alert-card warn">
  <b>⚠️ 资金高低切异动（单日）</b>
  {daily_rotation['text']}
</div>
""",
                unsafe_allow_html=True,
            )

        st.markdown(
            '<div style="margin-top:10px;font-size:.7rem;font-weight:600;color:#526176;">连续流入雷达（近3日每日净流入为正）</div>',
            unsafe_allow_html=True,
        )
        if consec_inflows:
            chips = []
            for item in consec_inflows:
                pct = item["pct"]
                hot = pct is not None and pct < 40
                label = item["name"] + (
                    f" 分位{pct:.0f}%" if pct is not None else ""
                ) + f" +{_fmt_compact_yi(item['total'])}"
                chips.append(f'<span class="chip {"hot" if hot else ""}">{label}</span>')
            st.markdown(f'<div class="chip-row">{"".join(chips)}</div>', unsafe_allow_html=True)
            st.caption("红框=估值分位<40%，具备“低估+连续流入”的新主线候选；仅观察，不追单日暴涨。")
        else:
            st.caption("暂无连续 3 日净流入的行业（样本不足或全行业轮动中）。")

        if "演示" in flow_view_meta.get("source", "") and mode == "auto":
            st.caption("实时资金流接口本次未取到数据，当前图表为演示兜底；点击侧边栏“刷新盘中数据”可重试。")
        elif period == "5日":
            st.caption(f"5 日资金口径 · {flow_view_meta['source']} · {flow_view_meta['asof']}")
        else:
            st.caption(f"今日资金口径 · {flow_view_meta['source']} · {flow_view_meta['asof']}")


    with st.container(border=True):
        st.markdown(
            """
<div class="section-kicker" style="margin-top:12px;">BROAD ETF / ANOMALY</div>
<div class="section-title" style="font-size:1.08rem;">宽基ETF大资金异动</div>
<div class="section-sub">放量≥1.5倍且收红 · 国家队护盘/建仓痕迹</div>
""",
            unsafe_allow_html=True,
        )
        etf_choice = st.radio(
            "宽基ETF",
            options=[item["short"] for item in BROAD_ETF_UNIVERSE],
            horizontal=True,
            label_visibility="collapsed",
            key="broad_etf_choice",
        )
        etf_item = next(
            (item for item in BROAD_ETF_UNIVERSE if item["short"] == etf_choice),
            BROAD_ETF_UNIVERSE[0],
        )
        etf_frame, etf_meta = etf_anomalies.get(
            etf_item["code"], (pd.DataFrame(), {"source": "", "asof": "", "is_demo": False})
        )
        if etf_frame.empty:
            err = (etf_meta.get("errors") or [""])[0]
            st.error(
                f"{etf_item['name']} 日线暂不可用：{_friendly_error(err)}。"
                "为避免误导，本图暂不显示，其余模块不受影响。"
            )
        else:
            esig = etf_anomaly_signal(etf_frame)
            icon = "🔴" if esig["count"] else "⚪"
            color = "#94600a" if esig["count"] else "#7c8797"
            st.markdown(
                f'<div class="margin-signal" style="color:{color};">'
                f"{icon} {esig['label']}</div>",
                unsafe_allow_html=True,
            )
            if go is not None:
                amounts = (etf_frame["成交额"] / 1e8).round(2)
                ma20 = (etf_frame["均额20"] / 1e8).round(2)
                chg = etf_frame["涨跌幅"].round(2)
                bar_colors = [
                    "#e23b3b" if hit else "#b9c3d1" for hit in etf_frame["异动"]
                ]
                hover = [
                    (
                        f"异动：放量{row['放量倍数']:.1f}倍，涨幅{row['涨跌幅']:+.2f}%"
                        if row["异动"]
                        else f"{row['日期']:%m-%d} 成交额 {row['成交额'] / 1e8:.2f}亿 · 涨跌 {row['涨跌幅']:+.2f}%"
                    )
                    for _, row in etf_frame.iterrows()
                ]
                fig = go.Figure()
                fig.add_bar(
                    x=etf_frame["日期"], y=amounts, marker_color=bar_colors,
                    name="成交额", text=hover, hovertemplate="%{text}<extra></extra>",
                )
                hit_rows = etf_frame[etf_frame["异动"]]
                if not hit_rows.empty:
                    # 异动柱上方直接标日期，避免不知道是哪一天
                    fig.add_scatter(
                        x=hit_rows["日期"], y=(hit_rows["成交额"] / 1e8) * 1.05,
                        mode="text",
                        text=[d.strftime("%m-%d") for d in hit_rows["日期"]],
                        textposition="top center",
                        textfont=dict(size=8, color="#c0392b"),
                        showlegend=False, hoverinfo="skip",
                    )
                fig.add_scatter(
                    x=etf_frame["日期"], y=ma20, mode="lines",
                    line=dict(color="#e0a800", width=2), name="20日均额",
                    hovertemplate="20日均额 %{y:.2f}亿<extra></extra>",
                )
                fig.add_scatter(
                    x=etf_frame["日期"], y=chg, mode="lines", yaxis="y2",
                    line=dict(color="#2f6fb0", width=1.6), name="涨跌幅",
                    hovertemplate="涨跌幅 %{y:+.2f}%<extra></extra>",
                )
                fig.update_layout(
                    height=210,
                    margin=dict(l=2, r=2, t=26, b=2),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="PingFang SC, Microsoft YaHei, sans-serif",
                              size=9, color="#4c5b70"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                                font=dict(size=8.5)),
                    xaxis=dict(showgrid=False, nticks=7, tickformat="%m-%d",
                               tickfont=dict(size=8.5)),
                    yaxis=dict(showgrid=True, gridcolor="#edf1f6",
                               tickfont=dict(size=8.5), title=dict(text="成交额(亿)", font=dict(size=8.5))),
                    yaxis2=dict(overlaying="y", side="right", showgrid=False,
                                tickfont=dict(size=8.5), zeroline=False,
                                title=dict(text="涨跌幅(%)", font=dict(size=8.5))),
                )
                _render_plotly(fig, config={"displayModeBar": False}, key="broad_etf_chart")
            st.caption(
                f"{etf_meta['source']} · {etf_meta['asof']} · "
                f"{etf_item['name']}（{etf_item['code']}）近30个交易日"
            )



    with st.container(border=True):
        st.markdown(
            """
<div class="section-kicker" style="margin-top:12px;">LEVERAGE / MARGIN</div>
<div class="section-title" style="font-size:1.08rem;">全市场两融水位</div>
<div class="section-sub">沪深两融余额近3个月 · 判断杠杆资金是否出清</div>
""",
            unsafe_allow_html=True,
        )
        if not margin_points:
            st.caption("两融数据暂不可用（交易所/金十接口限流），不影响其他模块；稍后点右上角刷新重试。")
        else:
            msig = margin_signal(margin_points)
            latest = margin_points[-1]
            base5 = margin_points[-6]["total"] if len(margin_points) >= 6 else margin_points[0]["total"]
            base20 = margin_points[-21]["total"] if len(margin_points) >= 21 else margin_points[0]["total"]
            chg5 = (latest["total"] / base5 - 1) * 100 if base5 else 0.0
            chg20 = (latest["total"] / base20 - 1) * 100 if base20 else 0.0
            icon = "🟡" if msig["level"] == "warn" else ("🟢" if msig["level"] == "good" else "⚪")
            st.markdown(
                f"""
<div class="margin-head">
  <span class="margin-value">{latest['total']:,.0f} 亿元</span>
  <span class="margin-chg">5日 {chg5:+.2f}% · 20日 {chg20:+.2f}%</span>
</div>
<div class="margin-signal {msig['level']}">{icon} {msig['label']}：{msig['detail']}</div>
""",
                unsafe_allow_html=True,
            )
            if go is not None:
                margin_values = [p["total"] for p in margin_points]
                low, high = min(margin_values), max(margin_values)
                pad = (high - low) * 0.12 or high * 0.01 or 1.0
                fig = go.Figure(
                    go.Scatter(
                        x=pd.to_datetime([p["date"] for p in margin_points]),
                        y=margin_values,
                        mode="lines",
                        line=dict(color="#2f6fb0", width=3),
                        hovertemplate="%{x|%m-%d}<br>两融余额 %{y:,.0f} 亿<extra></extra>",
                    )
                )
                fig.update_layout(
                    height=210,
                    margin=dict(l=2, r=6, t=8, b=2),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="PingFang SC, Microsoft YaHei, sans-serif", size=9, color="#4c5b70"),
                    showlegend=False,
                    xaxis=dict(showgrid=False, nticks=5, tickfont=dict(size=9)),
                    yaxis=dict(
                        showgrid=True, gridcolor="#edf1f6", tickfont=dict(size=9),
                        range=[low - pad, high + pad],
                    ),
                )
                _render_plotly(fig, config={"displayModeBar": False}, key="margin_chart")
            else:
                st.line_chart(
                    pd.DataFrame(
                        {"date": pd.to_datetime([p["date"] for p in margin_points]),
                         "total": [p["total"] for p in margin_points]}
                    ).set_index("date")["total"]
                )
            st.markdown(
                f'<div class="margin-alert {margin_alert(margin_points)["level"]}">'
                f'{margin_alert(margin_points)["text"]}</div>',
                unsafe_allow_html=True,
            )
            st.caption(
                f"{margin_meta['source']} · {margin_meta['asof']}"
                f"（{len(margin_points)} 个交易日，每日更新，不频繁请求）"
            )

# ----------------------------- 03/04 事件与 AI ---------------------------
with right_col:

    with st.container(border=True):
        st.markdown(
            """
<div class="section-kicker">03 / AI DECISION</div>
<div class="section-title" style="font-size:1.08rem;">AI 量化交易结论</div>
<div class="section-sub">剥离情绪 · 只输出条件、价位与动作</div>
""",
            unsafe_allow_html=True,
        )

        prompt = build_ai_prompt(
            valuation_rows=valuation_rows,
            inflow=inflow_today,
            outflow=outflow_today,
            supports=supports,
            holdings=holdings,
            macro=macro,
            msignal=msignal,
            consec_inflows=consec_inflows,
            etf_anomalies=etf_anomalies,
            margin_points=margin_points,
            total_capital=total_capital,
            tech_position=tech_position,
            cash=cash,
        )

        if api_key:
            try:
                with st.spinner("AI 正在生成今日结论…"):
                    conclusion = call_llm(
                        api_key,
                        base_url,
                        model or "deepseek-chat",
                        prompt,
                        st.session_state.ai_nonce,
                    )
                ai_source = f"{model or 'deepseek-chat'} · 大模型"
                ai_error = ""
            except Exception as exc:
                conclusion = build_local_conclusion(
                    valuation_rows, inflow_today, outflow_today, rotation,
                    supports, opportunities, holdings, resonance, events,
                    macro, msignal, consec_inflows,
                    tech_position, cash, rebound_take_profit,
                )
                ai_source = "本地规则引擎（大模型调用失败）"
                ai_error = _clip_error(exc, 180)
        else:
            conclusion = build_local_conclusion(
                valuation_rows, inflow_today, outflow_today, rotation,
                supports, opportunities, holdings, resonance, events,
                macro, msignal, consec_inflows,
                tech_position, cash, rebound_take_profit,
            )
            ai_source = "本地规则引擎（未配置 API Key）"
            ai_error = ""

        if ai_error:
            st.warning(f"大模型调用失败，已切换本地规则：{ai_error}")

        st.markdown(
            f"""
<div class="ai-panel">
  <div class="ai-meta">{ai_source}</div>
  <div class="ai-body">{_ai_html(conclusion)}</div>
  <div class="ai-local">结论来自规则/模型，不构成投资建议；破位纪律优先于任何观点。</div>
</div>
""",
            unsafe_allow_html=True,
        )

        button_col_a, button_col_b = st.columns([1, 1])
        with button_col_a:
            if st.button("重新生成结论", use_container_width=True):
                st.session_state.ai_nonce += 1
                st.rerun()
        with button_col_b:
            st.caption("AI 仅使用上表数据与事件，不读取未来函数。")

        if opportunity_note:
            st.caption(opportunity_note)
        else:
            st.caption("机会雷达：" + "；".join(
                f"{item['name']} PE分位{item['pct']:.1f}% / 3日{_fmt_yi(item['three_day'])}"
                for item in opportunities
            ))


    with st.container(border=True):
        st.markdown(
            """
<div class="section-kicker" style="margin-top:12px;">04 / EVENT COUNTDOWN</div>
<div class="section-title">宏观与产业事件倒计时</div>
<div class="section-sub">只保留可能引发暴涨/暴跌的硬核节点</div>
""",
            unsafe_allow_html=True,
        )
        event_html = []
        for event in events[:3]:
            days = _event_days(event)
            if days < 0:
                cls, day_text, day_unit, badge = "past", f"{abs(days)}", "天前", "已发生"
            elif days <= 1:
                cls, day_text, day_unit, badge = "urgent", f"{days}", "天", "红色预警 · 1 天内"
            elif days <= 3:
                cls, day_text, day_unit, badge = "warn", f"{days}", "天", "黄色提示 · 3 天内"
            else:
                cls, day_text, day_unit, badge = "normal", f"{days}", "天", f"距今 {days} 天"
            event_html.append(
                f"""
<div class="event-card {cls}">
  <div class="event-day">{day_text}<small>{day_unit}</small></div>
  <div class="event-body">
    <div class="event-meta">{event['date']} · {event['type']}</div>
    <div class="event-title">{event['title']}</div>
    <div class="event-impact">{event['impact']}</div>
    <span class="event-badge">{badge}</span>
  </div>
</div>
"""
            )
        st.markdown("".join(event_html), unsafe_allow_html=True)
        if event_source == "events.json":
            _ev_age = ""
            try:
                _ev_age = datetime.fromtimestamp(EVENTS_FILE.stat().st_mtime).strftime("%m-%d")
            except OSError:
                pass
            st.caption(f"事件日历自动更新（最后更新 {_ev_age}）· 每周一自动重建，也可手动刷新")
        else:
            st.caption("当前为内置事件；配置 Key 后每周自动生成，或点下方按钮立即更新")
        if api_key:
            if st.button("更新事件日历", use_container_width=True, key="refresh_events"):
                with st.spinner("DeepSeek 正在生成未来 15 天事件日历…"):
                    _new_events, _ev_err = generate_events(api_key, base_url, model)
                if _ev_err:
                    st.warning(_ev_err)
                else:
                    st.session_state.events_auto_done = True
                    st.session_state.events_auto_error = ""
                    st.rerun()
        if st.session_state.get("events_auto_error"):
            st.caption(f"自动更新失败：{st.session_state.events_auto_error}")

    with st.container(border=True):
        st.markdown(
            """
<div class="section-kicker">RISK LINE / SUPPORT</div>
<div class="section-title" style="font-size:1.08rem;">科技防线</div>
<div class="section-sub">20 日低点=第一支撑；跌破后减半，支撑下 5%=硬止损</div>
""",
            unsafe_allow_html=True,
        )
        support_html = []
        for row in supports:
            if row["dist"] <= 0:
                status, cls = "破位·执行减半", "broken"
            elif row["dist"] <= 3:
                status, cls = "贴近·盯紧", "near"
            else:
                status, cls = "安全", "safe"
            vr = row.get("vol_ratio")
            vr_txt = f"{vr:.2f}x" if vr is not None else "—"
            if vr is not None and vr >= 1.5:
                vr_txt += "放量"
            elif vr is not None and vr <= 0.7:
                vr_txt += "缩量"
            ma20_side = "上方" if row["above_ma20"] else "下方"
            support_html.append(
                f"""
<div class="support-row">
  <span class="support-name">{row['name']}</span>
  <span class="support-num">{row['latest']:.1f}</span>
  <span class="support-num">{row['low20']:.1f} · {row['dist']:+.1f}%</span>
  <span class="support-status {cls}">{status}</span>
</div>
<div class="support-sub">量比 {vr_txt} · MA20{ma20_side} {row['ma20_gap']:+.1f}% · MA20={row['ma20']:.1f}</div>
"""
            )
        st.markdown(
            '<div class="support-row" style="color:#8a95a5;font-size:.62rem;border-bottom:1px solid #edf1f6;">'
            '<span>指数</span><span>现价</span><span>支撑 / 距离</span><span>状态</span></div>'
            + "".join(support_html),
            unsafe_allow_html=True,
        )
        st.caption(
            f"5日线：科创50 {'站上' if supports[0]['above_ma5'] else '跌破'} · "
            f"科创芯片 {'站上' if supports[1]['above_ma5'] else '跌破'} · "
            f"通信 {'站上' if supports[2]['above_ma5'] else '跌破'}；"
            "未见放量右侧信号，深套仓位不补。"
        )

    with st.container(border=True):
        st.markdown(
            """
<div class="section-kicker" style="margin-top:12px;">MY POSITIONS</div>
<div class="section-title" style="font-size:1.08rem;">持仓盈亏监控</div>
<div class="section-sub">左侧边栏录入成本点位 · 跌幅≤-8% 触发硬止损参考</div>
""",
            unsafe_allow_html=True,
        )
        if holdings and tech_drain_text:
            st.markdown(
                f"""
<div class="alert-card trigger">
  <b>⚠️ 资金抽血警告</b><br>
  {tech_drain_text}，你的科技持仓正面临抛压，<b>请勿补仓</b>；反弹按纪律做T减仓。
</div>
""",
                unsafe_allow_html=True,
            )
        if not holdings:
            st.caption("未录入持仓。在左侧边栏「我的持仓」填写指数成本点位与市值后，这里会实时显示盈亏与止损信号。")
        else:
            holding_html = []
            for h in holdings:
                if h["pnl_pct"] <= -8:
                    badge, cls = "硬止损参考 · 减半", "broken"
                elif h["pnl_pct"] <= -3:
                    badge, cls = "警戒", "near"
                else:
                    badge, cls = "正常", "safe"
                holding_html.append(
                    f"""
<div class="support-row">
  <span class="support-name">{h['name']}</span>
  <span class="support-num">{h['pnl_pct']:+.1f}%</span>
  <span class="support-num">{h['cost']:.0f} → {h['latest']:.0f}</span>
  <span class="support-status {cls}">{badge}</span>
</div>
"""
                )
            st.markdown(
                '<div class="support-row" style="color:#8a95a5;font-size:.62rem;border-bottom:1px solid #edf1f6;">'
                '<span>持仓</span><span>盈亏</span><span>成本 → 现价</span><span>信号</span></div>'
                + "".join(holding_html),
                unsafe_allow_html=True,
            )
            total_pnl_amt = sum(h["amount"] * h["pnl_pct"] / 100 for h in holdings)
            st.caption(f"持仓浮动盈亏估算：{total_pnl_amt:+,.0f} 元（按指数点位线性估算，仅供参考）")

# ----------------------------- 02 资金流 ---------------------------------

# ---------------------------------------------------------------------------
# 底部说明
# ---------------------------------------------------------------------------

diag = " · ".join(
    [*(valuation_meta.get("errors") or []), *(flow_today_meta.get("errors") or [])]
)
st.markdown(
    f"""
<div class="footer">
数据链路：韭圈儿指数估值（PE/PB 近 5 年分位） · 中证指数官网（股息率） · AkShare / 东方财富（行业资金流·南向资金·全A宽度·ETF溢价率） · 新浪行情（美元/汇率/黄金） · 中债/FRED（美债名义与实际利率） · 本地事件字典。<br>
版本 {BUILD_TAG}　｜　估值兜底快照日期 {FALLBACK_ASOF}；当实时接口不可用时页面会明确标记“演示/快照”，不会伪装成实时数据。
事件与仓位参数可在左侧控制台及代码顶部调整。此工具用于个人研究与风险提醒，不构成投资建议。
</div>
""",
    unsafe_allow_html=True,
)

if diag:
    with st.expander("数据诊断（接口异常详情）", expanded=False):
        st.caption(" · ".join(_friendly_error(item) for item in diag.split(" · "))[:600])
