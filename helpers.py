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
                agriculture_structure_indicator TEXT, 
                basement_enclosure_crawlspace_type TEXT,
                condominium_coverage_type_code TEXT,
                policy_count INTEGER,
                crs_classification_code FLOAT,
                elevated_building_indicator TEXT,
                elevation_difference FLOAT,
                rated_flood_zone TEXT,
                house_worship TEXT,
                location_of_contents INTEGER,
                latitude FLOAT,
                longitude FLOAT,
                number_of_floors_in_the_insured_building INTEGER,
                non_profit_indicator TEXT,
                obstruction_type TEXT,
                occupancy_type TEXT,
                post_f_i_r_m_construction_indicator TEXT,
                original_construction_date INTEGER,
                reported_zip_code INTEGER,
                year_of_loss INTEGER,
                predicted_loss_ratio FLOAT,
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
                number_of_adresses INTEGER  NOT NULL DEFAULT 0
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
