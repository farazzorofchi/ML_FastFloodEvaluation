import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from flask import Flask, render_template, request, redirect, flash, session
from flask_session import Session
from sklearn.externals import joblib
from tempfile import mkdtemp
from werkzeug.security import check_password_hash, generate_password_hash

from helpers import open_db, apology, login_required


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
db = open_db()
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
            zip_agg['reported_zip_code'] == int(request.form.get("reported_zip_code"))
        ]
        zip_URL = f"static/{int(request.form.get('reported_zip_code'))}.png"
        df = (
            pd
            .DataFrame({
                'agriculture_structure_indicator': [request.form.get("agriculture_structure_indicator")],
                'basement_enclosure_crawlspace_type': [request.form.get("basement_enclosure_crawlspace_type")],
                'condominium_coverage_type_code': [request.form.get("condominium_coverage_type_code")],
                'policy_count': [request.form.get("policy_count")],
                'crs_classification_code': [request.form.get("crs_classification_code")],
                'elevated_building_indicator': [request.form.get("elevated_building_indicator")],
                'elevation_difference': [request.form.get("elevation_difference")],
                'rated_flood_zone': [request.form.get("rated_flood_zone")],
                'house_worship': [request.form.get("house_worship")],
                'latitude': [request.form.get("latitude")],
                'location_of_contents': [request.form.get("location_of_contents")],
                'longitude': [request.form.get("longitude")],
                'number_of_floors_in_the_insured_building': [request.form.get("number_of_floors_in_the_insured_building")],
                'non_profit_indicator': [request.form.get("non_profit_indicator")],
                'obstruction_type': [request.form.get("obstruction_type")],
                'occupancy_type': [request.form.get("occupancy_type")],
                'post_f_i_r_m_construction_indicator': [request.form.get("post_f_i_r_m_construction_indicator")],
                'year_of_loss': [request.form.get("year_of_loss")],
                'original_construction_date': [request.form.get("original_construction_date")],
                'reported_zip_code': [request.form.get("reported_zip_code")]
            })
            .astype({
                'basement_enclosure_crawlspace_type': 'float',
                'policy_count': 'float',
                'crs_classification_code': 'float',
                'elevation_difference': 'float',
                'latitude': 'float',
                'longitude': 'float',
                'number_of_floors_in_the_insured_building': 'float',
                'occupancy_type': 'float',
                'year_of_loss': 'int',
                'original_construction_date': 'int',
                'reported_zip_code': 'int',
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
                agriculture_structure_indicator, 
                basement_enclosure_crawlspace_type, 
                condominium_coverage_type_code, 
                policy_count, 
                crs_classification_code, 
                elevated_building_indicator, 
                elevation_difference, 
                rated_flood_zone, 
                house_worship, 
                location_of_contents, 
                latitude, 
                longitude, 
                number_of_floors_in_the_insured_building, 
                non_profit_indicator, 
                obstruction_type, 
                occupancy_type, 
                post_f_i_r_m_construction_indicator, 
                original_construction_date, 
                reported_zip_code, 
                year_of_loss,
                predicted_loss_ratio
            ) 
            VALUES (
                :user_id, 
                :agriculture_structure_indicator, 
                :basement_enclosure_crawlspace_type, 
                :condominium_coverage_type_code, 
                :policy_count, 
                :crs_classification_code, 
                :elevated_building_indicator, 
                :elevation_difference, 
                :rated_flood_zone, 
                :house_worship, 
                :location_of_contents, 
                :latitude, 
                :longitude, 
                :number_of_floors_in_the_insured_building, 
                :non_profit_indicator, 
                :obstruction_type, 
                :occupancy_type, 
                :post_f_i_r_m_construction_indicator, 
                :original_construction_date, 
                :reported_zip_code, 
                :year_of_loss,
                :predicted_loss_ratio
            )
            """,
            user_id=session["user_id"],
            agriculture_structure_indicator=request.form.get("agriculture_structure_indicator"),
            basement_enclosure_crawlspace_type=request.form.get("basement_enclosure_crawlspace_type"),
            condominium_coverage_type_code=request.form.get("condominium_coverage_type_code"),
            policy_count=request.form.get("policy_count"),
            crs_classification_code=request.form.get("crs_classification_code"),
            elevated_building_indicator=request.form.get("elevated_building_indicator"),
            elevation_difference=request.form.get("elevation_difference"),
            rated_flood_zone=request.form.get("rated_flood_zone"),
            house_worship=request.form.get("house_worship"),
            location_of_contents=request.form.get("location_of_contents"),
            latitude=request.form.get("latitude"),
            longitude=request.form.get("longitude"),
            number_of_floors_in_the_insured_building=request.form.get("number_of_floors_in_the_insured_building"),
            non_profit_indicator=request.form.get("non_profit_indicator"),
            obstruction_type=request.form.get("obstruction_type"),
            occupancy_type=request.form.get("occupancy_type"),
            post_f_i_r_m_construction_indicator=request.form.get("post_f_i_r_m_construction_indicator"),
            original_construction_date=request.form.get("original_construction_date"),
            reported_zip_code=request.form.get("reported_zip_code"),
            year_of_loss=request.form.get("year_of_loss"),
            predicted_loss_ratio=LR
        )
        db.execute(
            """
            UPDATE accounts 
            SET number_of_adresses = number_of_adresses + 1 
            WHERE uid = :user_id
            """,
            user_id=session["user_id"]
        )
        flash("Location Added!")
        ax = (
            sns.boxplot()
            if zip_temp.empty else
            sns.boxplot(
                x="reported_zip_code",
                y="loss_ratio_building",
                palette=["r", "g"],
                data=zip_temp
            )
        )
        ax.set_xlabel('Zip Code')
        ax.set_ylabel('Building Loss Ratio')
        fig = ax.get_figure()
        fig.set_size_inches(4.5, 5)
        fig.savefig(zip_URL)
        plt.close(fig)
        return render_template(
            "my_map.html",
            lat=request.form.get("latitude"),
            lon=request.form.get("longitude"),
            col=col,
            LR=LR,
            len_image=True if ax.lines else False,
            zip_URL=zip_URL,
            min_LR=round(zip_temp['loss_ratio_building'].min(), 2),
            max_LR=round(zip_temp['loss_ratio_building'].max(), 2),
            mean_LR=round(zip_temp['loss_ratio_building'].mean(), 2),
            len_zip=len(zip_temp),
            zip=request.form.get("reported_zip_code")
        )


@app.route("/login", methods=["GET", "POST"])
def login():
    session.clear()
    if not request.method == "POST":
        return render_template("login.html")
    else:
        potential_user = db.execute(
            """
            SELECT * 
            FROM accounts 
            WHERE username = :username
            """,
            username=request.form.get("username")
        )
        user = next(iter(potential_user), {})
        if not request.form.get("username"):
            return apology("must provide username", 403)
        elif not request.form.get("password"):
            return apology("must provide password", 403)
        elif not user:
            return apology("username does not exist", 403)
        elif not check_password_hash(user["hash"], request.form.get("password")):
            return apology("password does not match", 403)
        else:
            session["user_id"] = user["uid"]
            return render_template("input.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if not request.method == "POST":
        return render_template("register.html")    
    else:
        potential_user = db.execute(
            """
            SELECT * 
            FROM accounts 
            WHERE username = :username
            """,
            username=request.form.get("username")
        )            
        user = next(iter(potential_user), {})
        if not request.form.get("username"):
            return apology("must provide username", 400)
        elif not request.form.get("password"):
            return apology("must provide password", 400)
        elif request.form.get("password") != request.form.get("confirmation"):
            return apology("passwords do not match", 400)
        elif user:
            return apology("username already taken", 400)
        else:
            hash = generate_password_hash(request.form.get("password"))
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
                agriculture_structure_indicator, 
                basement_enclosure_crawlspace_type, 
                condominium_coverage_type_code, 
                policy_count, 
                crs_classification_code, 
                elevated_building_indicator, 
                elevation_difference, 
                rated_flood_zone, 
                house_worship, 
                location_of_contents, 
                latitude, 
                longitude, 
                number_of_floors_in_the_insured_building, 
                non_profit_indicator, 
                obstruction_type, 
                occupancy_type, 
                post_f_i_r_m_construction_indicator, 
                original_construction_date, 
                reported_zip_code, 
                date, 
                predicted_loss_ratio 
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
