import os
import requests
import sqlite3
import urllib.parse
from functools import wraps

from cs50 import SQL
from flask import redirect, render_template, session


def open_db():
    try:
        return SQL("sqlite:///users.db")
    except RuntimeError:
        conn = sqlite3.connect('users.db')
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE address (
                uid INTEGER, 
                agr TEXT, 
                basement TEXT,
                condo TEXT,
                policycount INTEGER,
                crsdiscount FLOAT,
                elevatedbuilding TEXT,
                elevationdifference FLOAT,
                floodzone TEXT,
                houseworship TEXT,
                locationofcontents INTEGER,
                latitude FLOAT,
                longitude FLOAT,
                numstories INTEGER,
                nonprofit TEXT,
                obstructiontype TEXT,
                occupancytype TEXT,
                postfirm TEXT,
                yearbuilt INTEGER,
                zipcode INTEGER,
                yearofloss INTEGER,
                lossratio FLOAT,
                date DATE NOT NULL DEFAULT CURRENT_DATE
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE accounts (
                uid INTEGER PRIMARY KEY AUTOINCREMENT ,
                username TEXT NOT NULL,
                hash TEXT NOT NULL,
                numaddresses INTEGER  NOT NULL DEFAULT 0
            );
            """
        )
        conn.commit()
        conn.close()
        return SQL("sqlite:///users.db")


def apology(message, code=400):
    """Render message as an apology to user."""
    def escape(s):
        """
        Escape special characters.

        https://github.com/jacebrowning/memegen#special-characters
        """
        for old, new in [
            ("-", "--"), (" ", "-"), ("_", "__"), ("?", "~q"),
            ("%", "~p"), ("#", "~h"), ("/", "~s"), ("\"", "''")
        ]:
            s = s.replace(old, new)
        return s
    return render_template("apology.html", top=code, bottom=escape(message)), code


def login_required(f):
    """
    Decorate routes to require login.

    http://flask.pocoo.org/docs/1.0/patterns/viewdecorators/
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if session.get("user_id") is None:
            return redirect("/login")
        return f(*args, **kwargs)
    return decorated_function


def lookup(symbol):
    """Look up quote for symbol."""

    # Contact API
    try:
        api_key = os.environ.get("API_KEY")
        response = requests.get(
            f"https://cloud-sse.iexapis.com/stable/stock/" +
            f"{urllib.parse.quote_plus(symbol)}/quote?token={api_key}"
        )
        response.raise_for_status()
    except requests.RequestException:
        return None

    # Parse response
    try:
        quote = response.json()
        return {
            "name": quote["companyName"],
            "price": float(quote["latestPrice"]),
            "symbol": quote["symbol"]
        }
    except (KeyError, TypeError, ValueError):
        return None


def usd(value):
    """Format value as USD."""
    return f"${value:,.2f}"
