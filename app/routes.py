from flask import render_template,redirect,request,flash,session,url_for
from flask_login import logout_user,current_user, login_user, login_required
from app import app,db
from app.models import User, MessageData
from datetime import datetime
from app.core.algo import *
import plotly.express as px


@app.route('/')
@app.route('/index')
@login_required
def index():
    return render_template('index.html',title='home')

@app.route('/login',methods=['GET', 'POST'])
def login():
    if request.method=='POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username and password:
            user = User.query.filter_by(username=username).first()
            if user is None or not user.check_password(password):
                flash('Invalid username or password','danger')
                return redirect(url_for('login'))
            login_user(user, remember=True)
            return redirect(url_for('index'))
    return render_template('login.html', title='Sign In')

    
@app.route('/register',methods=['GET', 'POST'])
def register():
    if request.method=='POST':
        email = request.form.get('email')
        username = request.form.get('username')
        cpassword = request.form.get('cpassword')
        password = request.form.get('password')
        print(cpassword, password, cpassword==password)
        if username and password and cpassword and email:
            if cpassword != password:
                flash('Password do not match','danger')
                return redirect('/register')
            else:
                if User.query.filter_by(email=email).first() is not None:
                    flash('Please use a different email address','danger')
                    return redirect('/register')
                elif User.query.filter_by(username=username).first() is not None:
                    flash('Please use a different username','danger')
                    return redirect('/register')
                else:
                    user = User(username=username, email=email)
                    user.set_password(password)
                    db.session.add(user)
                    db.session.commit()
                    flash('Congratulations, you are now a registered user!','success')
                    return redirect(url_for('login'))
        else:
            flash('Fill all the fields','danger')
            return redirect('/register')

    return render_template('register.html', title='Sign Up page')


@app.route('/forgot',methods=['GET', 'POST'])
def forgot():
    if request.method=='POST':
        email = request.form.get('email')
        if email:
            pass
    return render_template('forgot.html', title='Password reset page')
    

@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('index'))

@login_required
@app.route('/user/<username>')
def user(username):
    user = User.query.filter_by(username=username).first_or_404()
    return render_template('profile.html', user=user, title=f'{user.username} profile')

@app.route('/about')
def about():
    return render_template('about.html')

@app.before_request
def before_request():
    if current_user.is_authenticated:
        current_user.last_seen = datetime.utcnow()
        db.session.commit()

@app.route('/edit_profile', methods=['GET', 'POST'])
@login_required
def edit_profile():
    if request.method=='POST':
        current_user.username = request.form.get('username')
        current_user.about_me = request.form.get('aboutme')
        db.session.commit()
        flash('Your changes have been saved.','success')
        return redirect(url_for('edit_profile'))
    return render_template('edit_profile.html', title='Edit Profile',user=user)

@app.route('/mbo', methods=['GET','POST'])
def select_options():
    countries=df['Country'].unique().tolist()
    return render_template('choose_option.html',countries=countries) 

@app.route('/results',methods=["POST","GET"])
def results():
    if request.method == "POST":
        country = request.form.get('country')
        confidence = int(request.form.get('confidence'))
        lift = int(request.form.get('lift'))
        support = int(request.form.get('support'))
        length = int(request.form.get('length'))
        support /= 100
        lift /= 100
        confidence /= 100
        print("country name",country)
        print("min support",support)
        print("min lift",lift)
        print("min confidence",confidence)
        print("max length",length)
        rules = generate_basket(df, country=country, min_support=support, max_length=length)
        results = get_rules(rules)
        results = results.head(25).reset_index()
        fig1 = px.scatter_3d(data_frame=results,x=results.index,y='confidence',z='lift',title="visualization b/w products",width=500, height=500)
        fig2 = px.histogram(data_frame=results,x=results.index,y='confidence',title="visualization b/w products",width=500, height=500,marginal='box')
        g1 = fig1.to_json()
        g2 = fig2.to_json()
        session['g1'] =g1
        session['g2'] =g2
        return render_template('results.html',data=results.head(100).to_html(),c=confidence,l=lift,s=support,ml =length)
    else:
        return redirect('/mbo')


@app.route('/visualize')
def vis():
    if 'g1' in session and 'g2' in session:
        return render_template('vis.html')
    else:
        return redirect('/mbo')

filepath = 'app\\core\\cleaned_online_retail.csv'
print("loading the market basket dataset")
df = load_file(filepath)
print("processed dataset, reload home page")