import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from cs50 import SQL
from flask import Flask, render_template, request, redirect, flash, session
from flask_session import Session
from sklearn.externals import joblib
from tempfile import mkdtemp
from werkzeug.security import check_password_hash, generate_password_hash

from helpers import apology, login_required


app = Flask(__name__)
# Ensure templates are auto-reloaded
app.config["TEMPLATES_AUTO_RELOAD"] = True


@app.after_request
def after_request(response):
    """Ensure responses aren't cached"""
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Expires"] = 0
    response.headers["Pragma"] = "no-cache"
    return response


# Configure session to use filesystem (instead of signed cookies)
app.config["SESSION_FILE_DIR"] = mkdtemp()
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"


Session(app)
db = SQL("sqlite:///users.db")
zip_agg = pd.read_csv("Zip_Aggregate.csv")
loaded_model = joblib.load('str_model.pkl')


@app.route("/")
@login_required
def index():
    return render_template("input.html")


@app.route("/input", methods=["GET", "POST"])
@login_required
def input():
    if not request.method == "POST":
        return render_template("input.html")
    else:
        zip_temp = zip_agg[
            zip_agg['ZipCode'] == int(request.form.get("zipcode"))
        ]
        len_zip = len(zip_temp)
        min_LR = round(zip_temp['loss_ratio_building'].min(), 2)
        max_LR = round(zip_temp['loss_ratio_building'].max(), 2)
        mean_LR = round(zip_temp['loss_ratio_building'].mean(), 2)
        if not len_zip > 1:
            len_image = False
            zip_URL = ""
        else:
            len_image = True
            zip_URL = f"static/{int(request.form.get('zipcode'))}.png"
            ax = sns.boxplot(
                x="ZipCode",
                y="loss_ratio_building",
                palette=["r", "g"],
                data=zip_agg[
                    zip_agg['ZipCode'] == int(request.form.get("zipcode"))
                ]
            )
            ax.set_xlabel('Zip Code')
            ax.set_ylabel('Building Loss Ratio')
            fig = ax.get_figure()
            fig.set_size_inches(4.5, 5)
            fig.savefig(zip_URL)
            plt.close(fig)
        df = (
            pd
            .DataFrame({
                'agriculturestructureindicator': [request.form.get("agriculture")],
                'basementenclosurecrawlspacetype': [request.form.get("basement")],
                'condominiumindicator': [request.form.get("condominium")],
                'policycount': [request.form.get("policycount")],
                'crsdiscount': [request.form.get("crsdiscount")],
                'elevatedbuildingindicator': [request.form.get("elevatedbuilding")],
                'elevationdifference': [request.form.get("elevationdifference")],
                'floodzone': [request.form.get("floodzone")],
                'houseworship': [request.form.get("houseworship")],
                'latitude': [request.form.get("latitude")],
                'locationofcontents': [request.form.get("locationofcontents")],
                'longitude': [request.form.get("longitude")],
                'numberoffloorsintheinsuredbuilding': [request.form.get("numberofstories")],
                'nonprofitindicator': [request.form.get("nonprofit")],
                'obstructiontype': [request.form.get("obstruction")],
                'occupancytype': [request.form.get("occupancy")],
                'postfirmconstructionindicator': [request.form.get("postfirm")],
                'yearofloss': [request.form.get("damageyear")],
                'yearbuilt': [request.form.get("yearbuilt")],
                'ZipCode': [request.form.get("zipcode")]
            })
            .astype({
                'basementenclosurecrawlspacetype': 'float',
                'policycount': 'float',
                'crsdiscount': 'float',
                'elevationdifference': 'float',
                'latitude': 'float',
                'longitude': 'float',
                'numberoffloorsintheinsuredbuilding': 'float',
                'occupancytype': 'float',
                'yearofloss': 'int',
                'yearbuilt': 'int',
                'ZipCode': 'int',
            })
        )
        LR = float(round(loaded_model.predict(df)[0], 2))
        if LR <= 0.05:
            col = "green"
        elif (LR > 0.05) and (LR <= 0.2):
            col = "yellow"
        else:
            col = "red"
        db.execute(
            """
            INSERT INTO address (
                uid, 
                agr, 
                basement, 
                condo, 
                policycount, 
                crsdiscount, 
                elevatedbuilding, 
                elevationdifference, 
                floodzone, 
                houseworship, 
                locationofcontents, 
                latitude, 
                longitude, 
                numstories, 
                nonprofit, 
                obstructiontype, 
                occupancytype, 
                postfirm, 
                yearbuilt, 
                zipcode, 
                yearofloss,
                lossratio
            ) 
            VALUES (
                :user_id, 
                :agriculture, 
                :basement, 
                :condominium, 
                :policycount, 
                :crsdiscount, 
                :elevatedbuilding, 
                :elevationdifference, 
                :floodzone, 
                :houseworship, 
                :locationofcontents, 
                :latitude, 
                :longitude, 
                :numstories, 
                :nonprofit, 
                :obstructiontype, 
                :occupancytype, 
                :postfirm, 
                :yearbuilt, 
                :zipcode, 
                :yearofloss,
                :lossratio
            )
            """,
            user_id=session["user_id"],
            agriculture=request.form.get("agriculture"),
            basement=request.form.get("basement"),
            condominium=request.form.get("condominium"),
            policycount=request.form.get("policycount"),
            crsdiscount=request.form.get("crsdiscount"),
            elevatedbuilding=request.form.get("elevatedbuilding"),
            elevationdifference=request.form.get("elevationdifference"),
            floodzone=request.form.get("floodzone"),
            houseworship=request.form.get("houseworship"),
            locationofcontents=request.form.get("locationofcontents"),
            latitude=request.form.get("latitude"),
            longitude=request.form.get("longitude"),
            numstories=request.form.get("numberofstories"),
            nonprofit=request.form.get("nonprofit"),
            obstructiontype=request.form.get("obstruction"),
            occupancytype=request.form.get("occupancy"),
            postfirm=request.form.get("postfirm"),
            yearbuilt=request.form.get("yearbuilt"),
            zipcode=request.form.get("zipcode"),
            yearofloss=request.form.get("damageyear"),
            lossratio=LR
        )
        db.execute(
            """
            UPDATE accounts 
            SET numaddresses = numaddresses + 1 
            WHERE uid = :user_id
            """,
            user_id=session["user_id"]
        )
        flash("Location Added!")
        return render_template(
            "my_map.html",
            lat=request.form.get("latitude"),
            lon=request.form.get("longitude"),
            col=col,
            LR=LR,
            len_image=len_image,
            zip_URL=zip_URL,
            min_LR=min_LR,
            max_LR=max_LR,
            mean_LR=mean_LR,
            len_zip=len_zip,
            zip=request.form.get("zipcode")
        )


@app.route("/login", methods=["GET", "POST"])
def login():
    session.clear()
    if not request.method == "POST":
        return render_template("login.html")
    else:
        if not request.form.get("username"):
            return apology("must provide username", 403)
        elif not request.form.get("password"):
            return apology("must provide password", 403)
        else:
            rows = db.execute(
                """
                SELECT * 
                FROM accounts 
                WHERE username = :username
                """,
                username=request.form.get("username")
            )
            if (
                (len(rows) != 1) or 
                (not check_password_hash(rows[0]["hash"], request.form.get("password")))
            ):
                return apology("invalid username and/or password", 403)
            else:
                session["user_id"] = rows[0]["uid"]
                return render_template("input.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if not request.method == "POST":
        return render_template("register.html")    
    else:
        if not request.form.get("username"):
            return apology("Must provide username", 400)
        elif not request.form.get("password"):
            return apology("Must provide password", 400)
        elif request.form.get("password") != request.form.get("confirmation"):
            return apology("Password do not match", 400)
        else:
            hash = generate_password_hash(request.form.get("password"))
            check_user_id = db.execute(
                """
                SELECT * 
                FROM accounts 
                WHERE username = :username
                """,
                username=request.form.get("username")
            )            
            if len(check_user_id) >= 1:
                return apology("username taken", 400)
            else:
                new_user_id = db.execute(
                    """
                    INSERT INTO accounts (username, hash) 
                    VALUES (:username, :hash)
                    """,
                    username=request.form.get("username"),
                    hash=hash
                )
                session["user_id"] = new_user_id
                flash("Registered")
                return render_template("input.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")


@app.route("/history", methods=["GET", "POST"])
@login_required
def history():
    """Show history of transactions"""
    if request.method == "GET":
        transactions = db.execute(
            """
            SELECT 
                agr, 
                basement, 
                condo, 
                policycount, 
                crsdiscount, 
                elevatedbuilding, 
                elevationdifference, 
                floodzone, 
                houseworship, 
                locationofcontents, 
                latitude, 
                longitude, 
                numstories, 
                nonprofit, 
                obstructiontype, 
                occupancytype, 
                postfirm, 
                yearbuilt, 
                zipcode, 
                date, 
                lossratio 
            FROM address 
            WHERE uid = :user_id 
            ORDER BY date ASC
            """,
            user_id=session["user_id"]
        )
        return render_template("history.html", transactions=transactions)
    else:
        db.execute(
            """
            DELETE 
            FROM address 
            WHERE uid = :user_id
            """,
            user_id=session["user_id"]
        )
        return render_template("history.html")


@app.route('/about', methods=["GET"])
@login_required
def about():
    return render_template("about.html")


@app.route('/input_information', methods=["GET"])
@login_required
def input_information():
    return render_template("info.html")


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 33507))
    app.run(port=port)
