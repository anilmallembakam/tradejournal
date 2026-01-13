import streamlit as st
import pandas as pd
from datetime import datetime, date, timezone
from dateutil import parser
from supabase import create_client, Client

st.set_page_config(page_title="Trading Journal + Risk Guard (MVP)", layout="wide")

# ----------------------------
# Config / Supabase client
# ----------------------------
def get_supabase() -> Client:
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_ANON_KEY"]
    return create_client(url, key)

sb = get_supabase()

# ----------------------------
# Helpers
# ----------------------------
def money(x):
    try:
        return f"${float(x):,.2f}"
    except Exception:
        return str(x)

def parse_dt(x):
    # Accept many formats; store as ISO (timestamptz)
    if pd.isna(x):
        return None
    if isinstance(x, (datetime,)):
        dt = x
    else:
        dt = parser.parse(str(x))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()

def calc_pnl(row):
    # PnL for completed trade.
    # BUY then SELL: (exit - entry) * qty - fees
    # SELL then BUY (short): (entry - exit) * qty - fees
    qty = float(row["qty"])
    entry = float(row["entry_price"])
    exitp = float(row["exit_price"])
    fees = float(row.get("fees", 0) or 0)

    side = str(row["side"]).upper().strip()
    if side not in ("BUY", "SELL"):
        side = "BUY"

    if side == "BUY":
        pnl = (exitp - entry) * qty - fees
    else:
        pnl = (entry - exitp) * qty - fees
    return pnl

def require_login():
    if "session" not in st.session_state or st.session_state.session is None:
        st.warning("Please log in to continue.")
        auth_ui()
        st.stop()

def auth_ui():
    st.title("Trading Journal + Risk Guard (MVP)")
    st.caption("Login / Signup (Supabase Auth)")

    tab1, tab2 = st.tabs(["Log in", "Sign up"])

    with tab1:
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_pw")
        if st.button("Log in", type="primary"):
            try:
                res = sb.auth.sign_in_with_password({"email": email, "password": password})
                st.session_state.session = res.session
                st.session_state.user = res.user
                st.success("Logged in!")
                st.rerun()
            except Exception as e:
                st.error(f"Login failed: {e}")

    with tab2:
        email2 = st.text_input("Email", key="signup_email")
        password2 = st.text_input("Password (min 6 chars)", type="password", key="signup_pw")
        if st.button("Create account"):
            try:
                res = sb.auth.sign_up({"email": email2, "password": password2})
                st.info("Account created. If email confirmation is enabled, check your inbox.")
                st.info("Now log in from the Log in tab.")
            except Exception as e:
                st.error(f"Signup failed: {e}")

def logout_button():
    if st.button("Log out"):
        try:
            sb.auth.sign_out()
        except Exception:
            pass
        st.session_state.session = None
        st.session_state.user = None
        st.rerun()

def ensure_rules_row(user_id: str):
    # Create default row if missing
    existing = sb.table("risk_rules").select("*").eq("user_id", user_id).execute()
    if not existing.data:
        sb.table("risk_rules").insert({
        "daily_max_loss": 100,
        "daily_max_trades": 5,
        "max_risk_per_trade": 50,
        "cooldown_after_losses": 2
        }).execute()


def get_rules(user_id: str):
    ensure_rules_row(user_id)
    res = sb.table("risk_rules").select("*").eq("user_id", user_id).single().execute()
    return res.data

def update_rules(user_id: str, daily_max_loss, daily_max_trades, max_risk_per_trade, cooldown_after_losses):
    sb.table("risk_rules").update({
        "daily_max_loss": float(daily_max_loss),
        "daily_max_trades": int(daily_max_trades),
        "max_risk_per_trade": float(max_risk_per_trade),
        "cooldown_after_losses": int(cooldown_after_losses),
        "updated_at": datetime.now(timezone.utc).isoformat()
    }).eq("user_id", user_id).execute()

def fetch_trades(user_id: str, start_iso=None, end_iso=None):
    q = sb.table("trades").select("*").eq("user_id", user_id).order("trade_datetime", desc=True)
    if start_iso:
        q = q.gte("trade_datetime", start_iso)
    if end_iso:
        q = q.lt("trade_datetime", end_iso)
    res = q.execute()
    return res.data or []

def compute_today_metrics(trades_today: pd.DataFrame):
    if trades_today.empty:
        return 0.0, 0, 0  # pnl, trade_count, loss_streak

    pnl = float(trades_today["pnl"].sum())
    trade_count = int(len(trades_today))

    # loss streak: count consecutive losses from latest backwards
    df = trades_today.sort_values("trade_datetime", ascending=False)
    streak = 0
    for x in df["pnl"].tolist():
        if float(x) < 0:
            streak += 1
        else:
            break
    return pnl, trade_count, streak

# ----------------------------
# App
# ----------------------------
if "session" not in st.session_state:
    st.session_state.session = None
    st.session_state.user = None

if st.session_state.session is None:
    auth_ui()
    st.stop()

user_id = st.session_state.user.id
st.sidebar.success(f"Logged in as {st.session_state.user.email}")
logout_button()

# Navigation
page = st.sidebar.radio("Navigate", ["Dashboard", "Import Trades (CSV)", "Risk Rules", "Trades"])

# Date range helpers for today
today = date.today()
start_today = datetime(today.year, today.month, today.day, tzinfo=timezone.utc).isoformat()
start_tomorrow = datetime(today.year, today.month, today.day, tzinfo=timezone.utc).replace(day=today.day)  # placeholder
start_tomorrow = (datetime(today.year, today.month, today.day, tzinfo=timezone.utc) + pd.Timedelta(days=1)).isoformat()

rules = get_rules(user_id)

# ----------------------------
# Dashboard
# ----------------------------
if page == "Dashboard":
    st.title("Dashboard")

    trades = fetch_trades(user_id, start_iso=start_today, end_iso=start_tomorrow)
    df = pd.DataFrame(trades)

    if df.empty:
        st.info("No trades found for today. Import a CSV to start.")
        st.stop()

    # Normalize types
    df["trade_datetime"] = pd.to_datetime(df["trade_datetime"], utc=True)
    df["pnl"] = pd.to_numeric(df["pnl"], errors="coerce").fillna(0)

    today_pnl, trade_count, loss_streak = compute_today_metrics(df)

    daily_max_loss = float(rules["daily_max_loss"])
    daily_max_trades = int(rules["daily_max_trades"])
    cooldown_after_losses = int(rules["cooldown_after_losses"])

    # Risk status
    loss_limit_hit = today_pnl <= -abs(daily_max_loss)
    trades_limit_hit = trade_count >= daily_max_trades
    cooldown_hit = loss_streak >= cooldown_after_losses

    status = "OK"
    if loss_limit_hit or trades_limit_hit or cooldown_hit:
        status = "LOCKED"
    elif (today_pnl <= -0.7 * abs(daily_max_loss)) or (trade_count >= int(0.7 * daily_max_trades)):
        status = "WARNING"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Today's P/L", money(today_pnl))
    c2.metric("Trades Today", trade_count)
    c3.metric("Loss Streak", loss_streak)
    c4.metric("Risk Status", status)

    # Progress bars
    st.subheader("Risk Meter")
    loss_progress = min(1.0, max(0.0, (-today_pnl) / abs(daily_max_loss))) if daily_max_loss != 0 else 0.0
    trades_progress = min(1.0, trade_count / max(1, daily_max_trades))

    st.write("Daily loss usage")
    st.progress(loss_progress)
    st.write("Daily trade count usage")
    st.progress(trades_progress)

    if status == "LOCKED":
        reasons = []
        if loss_limit_hit:
            reasons.append(f"Daily max loss hit ({money(daily_max_loss)})")
        if trades_limit_hit:
            reasons.append(f"Daily max trades hit ({daily_max_trades})")
        if cooldown_hit:
            reasons.append(f"Cooldown triggered (loss streak ≥ {cooldown_after_losses})")
        st.error("LOCKED — Stop trading for today.\n\n" + "\n".join([f"- {r}" for r in reasons]))

    st.subheader("Today’s trades")
    show_cols = ["trade_datetime", "symbol", "side", "qty", "entry_price", "exit_price", "fees", "pnl"]
    st.dataframe(df[show_cols].sort_values("trade_datetime", ascending=False), use_container_width=True)

# ----------------------------
# Import CSV
# ----------------------------
elif page == "Import Trades (CSV)":
    st.title("Import Trades (CSV)")
    st.caption("Upload any broker CSV. You'll map columns once and import completed trades.")

    up = st.file_uploader("Upload CSV", type=["csv"])
    if not up:
        st.info("Upload a CSV to continue.")
        st.stop()

    raw = pd.read_csv(up)
    st.write("Preview:")
    st.dataframe(raw.head(20), use_container_width=True)

    st.subheader("Map your columns")
    cols = list(raw.columns)

    col_dt = st.selectbox("Trade date/time column", cols)
    col_symbol = st.selectbox("Symbol column", cols)
    col_side = st.selectbox("Side column (BUY/SELL) — optional", ["(none)"] + cols)
    col_qty = st.selectbox("Quantity column", cols)
    col_entry = st.selectbox("Entry price column", cols)
    col_exit = st.selectbox("Exit price column", cols)
    col_fees = st.selectbox("Fees column — optional", ["(none)"] + cols)

    default_side = st.selectbox("If no Side column, assume:", ["BUY", "SELL"], index=0)

    if st.button("Import trades", type="primary"):
        df = raw.copy()

        # Build normalized df
        out = pd.DataFrame()
        out["trade_datetime"] = df[col_dt].apply(parse_dt)
        out["symbol"] = df[col_symbol].astype(str).str.upper().str.strip()

        if col_side != "(none)":
            out["side"] = df[col_side].astype(str).str.upper().str.strip()
            out["side"] = out["side"].where(out["side"].isin(["BUY","SELL"]), default_side)
        else:
            out["side"] = default_side

        out["qty"] = pd.to_numeric(df[col_qty], errors="coerce")
        out["entry_price"] = pd.to_numeric(df[col_entry], errors="coerce")
        out["exit_price"] = pd.to_numeric(df[col_exit], errors="coerce")

        if col_fees != "(none)":
            out["fees"] = pd.to_numeric(df[col_fees], errors="coerce").fillna(0)
        else:
            out["fees"] = 0.0

        # Drop bad rows
        out = out.dropna(subset=["trade_datetime", "symbol", "qty", "entry_price", "exit_price"])

        # Compute pnl
        out["pnl"] = out.apply(calc_pnl, axis=1)

        # Insert rows
        rows = []
        for _, r in out.iterrows():
            rows.append({
                "user_id": user_id,
                "trade_datetime": r["trade_datetime"],
                "symbol": r["symbol"],
                "side": r["side"],
                "qty": float(r["qty"]),
                "entry_price": float(r["entry_price"]),
                "exit_price": float(r["exit_price"]),
                "fees": float(r["fees"]),
                "pnl": float(r["pnl"]),
            })

        if not rows:
            st.warning("No valid rows found after mapping. Check your column selections.")
            st.stop()

        try:
            sb.table("trades").insert(rows).execute()
            st.success(f"Imported {len(rows)} trades.")
            st.info("Go to Dashboard to see Risk Guard + stats.")
        except Exception as e:
            st.error(f"Import failed: {e}")

# ----------------------------
# Risk Rules
# ----------------------------
elif page == "Risk Rules":
    st.title("Risk Rules")
    st.caption("These rules power the Risk Guard warnings & lock.")

    daily_max_loss = st.number_input("Daily Max Loss ($)", min_value=1.0, value=float(rules["daily_max_loss"]), step=10.0)
    daily_max_trades = st.number_input("Daily Max Trades", min_value=1, value=int(rules["daily_max_trades"]), step=1)
    max_risk_per_trade = st.number_input("Max Risk Per Trade ($) (placeholder for later features)", min_value=1.0, value=float(rules["max_risk_per_trade"]), step=5.0)
    cooldown_after_losses = st.number_input("Cooldown after consecutive losses", min_value=1, value=int(rules["cooldown_after_losses"]), step=1)

    if st.button("Save rules", type="primary"):
        try:
            update_rules(user_id, daily_max_loss, daily_max_trades, max_risk_per_trade, cooldown_after_losses)
            st.success("Rules saved.")
        except Exception as e:
            st.error(f"Save failed: {e}")

# ----------------------------
# Trades page
# ----------------------------
elif page == "Trades":
    st.title("Trades")
    st.caption("All imported trades (latest first).")

    trades = fetch_trades(user_id)
    df = pd.DataFrame(trades)
    if df.empty:
        st.info("No trades yet. Import a CSV to start.")
        st.stop()

    df["trade_datetime"] = pd.to_datetime(df["trade_datetime"], utc=True)
    df["pnl"] = pd.to_numeric(df["pnl"], errors="coerce").fillna(0)

    st.dataframe(
        df[["trade_datetime","symbol","side","qty","entry_price","exit_price","fees","pnl"]]
        .sort_values("trade_datetime", ascending=False),
        use_container_width=True
    )

    st.subheader("Quick stats")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total P/L", money(df["pnl"].sum()))
    c2.metric("Win rate", f"{(df['pnl'] > 0).mean()*100:.1f}%")
    avg = df["pnl"].mean() if len(df) else 0
    c3.metric("Avg trade P/L", money(avg))

