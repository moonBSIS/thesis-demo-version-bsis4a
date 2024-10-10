from flask import Flask, render_template, request, flash, redirect, session, url_for, jsonify, send_file
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import distinct
from datetime import datetime, timedelta
from flask_session import Session
from flask_migrate import Migrate
from flask_wtf import FlaskForm
from wtforms import StringField, IntegerField, SubmitField, HiddenField, SelectField, PasswordField, DateField
from wtforms.validators import DataRequired, Length, EqualTo
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy.exc import IntegrityError
import base64
import os
import numpy as np
import joblib
from flask_bootstrap import Bootstrap
from flask_login import login_required, current_user

app = Flask(__name__, static_folder='static')
bootstrap = Bootstrap(app)

UPLOAD_FOLDER = 'static/images'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}


app.config['SECRET_KEY'] = '4046bde895cc19ca9cbd373a'
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql+psycopg2://postgres:1234@localhost/malnutritiondb3'
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_TYPE'] = 'filesystem'
app.config['UPLOAD_FOLDER'] = 'static/images'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER) 

db = SQLAlchemy(app)
migrate = Migrate(app, db)
Session(app)


# Load the logistic regression model
model = joblib.load('logistic_regression.pkl')
barangays = ['Aganan', 'Amparo', 'Anilao', 'Amparo', 'Balabag', 'Cabugao Norte', 'Cabugao Sur', 'Jibao-an', 'Mali-ao', 'Pagsanga-an', 'Pal-agon',
             'Pandac', 'Purok 1', 'Purok 2', 'Purok 3', 'Purok 4', 'Tigum', 'Ungka 1', 'Ungka 2']

admin_credentials = {
    1: {'username': 'admin1', 'password': 'password1'},
    2: {'username': 'admin_amparo', 'password': 'password2'},
    3: {'username': 'admin_anilao', 'password': 'password3'},
    4: {'username': 'admin_balabag', 'password': 'password4'},
    5: {'username': 'admin_cabugao_norte', 'password': 'password5'},
    6: {'username': 'admin_cabugao_sur', 'password': 'password6'},
    7: {'username': 'admin_jibao_an', 'password': 'password7'},
    8: {'username': 'admin_mali_ao', 'password': 'password8'},
    9: {'username': 'admin_pagsanga_an', 'password': 'password9'},
    10: {'username': 'admin_pal_agon', 'password': 'password10'},
    11: {'username': 'admin_pandac', 'password': 'password11'},
    12: {'username': 'admin_purok_1', 'password': 'password12'},
    13: {'username': 'admin_purok_2', 'password': 'password13'},
    14: {'username': 'admin_purok_3', 'password': 'password14'},
    15: {'username': 'admin_purok_4', 'password': 'password15'},
    16: {'username': 'admin_tigum', 'password': 'password16'},
    17: {'username': 'admin_ungka_1', 'password': 'password17'},
    18: {'username': 'admin_ungka_2', 'password': 'password18'},
}

# Admin Authentication
def authenticate_admin(username, password):
    for admin_id, creds in admin_credentials.items():
        if creds['username'] == username and creds['password'] == password:
            return admin_id
    return None


# User Class
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    barangay = db.Column(db.String(50), nullable=False)
    households = db.relationship('Household', back_populates='user', lazy='dynamic')

    def set_password(self, password):
        self.password = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password, password)

    def __repr__(self):
        return f'<User {self.username}>'

# Child Class
class Child(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    first_name = db.Column(db.String(50), nullable=False)
    last_name = db.Column(db.String(50), nullable=False)
    parent_id = db.Column(db.Integer, nullable=False)

    def __repr__(self):
        return f'<Child {self.first_name} {self.last_name}>'

class ChildForm(FlaskForm):
    first_name = StringField('First Name', validators=[DataRequired()])
    last_name = StringField('Last Name', validators=[DataRequired()])
    submit = SubmitField('Submit')

# Household Class
class Household(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    address = db.Column(db.String(100), nullable=False)
    mother_first_name = db.Column(db.String(100), nullable=False)
    mother_last_name = db.Column(db.String(100), nullable=False)
    father_first_name = db.Column(db.String(100), nullable=False)
    father_last_name = db.Column(db.String(100), nullable=False)
    mother_age = db.Column(db.Integer, nullable=False)
    father_age = db.Column(db.Integer, nullable=False)

    user = db.relationship('User', back_populates='households')

    def __repr__(self):
        return f'<Household {self.id}>'
    
class HouseholdForm(FlaskForm):
    address = StringField('Address', validators=[DataRequired(), Length(max=100)])
    mother_first_name = StringField('Mother First Name', validators=[DataRequired(), Length(max=100)])
    mother_last_name = StringField('Mother Last Name', validators=[DataRequired(), Length(max=100)])
    father_first_name = StringField('Father First Name', validators=[DataRequired(), Length(max=100)])
    father_last_name = StringField('Father Last Name', validators=[DataRequired(), Length(max=100)])
    mother_age = IntegerField('Mother Age', validators=[DataRequired()])
    father_age = IntegerField('Father Age', validators=[DataRequired()])
    submit = SubmitField('Save')

# Prediction Class
class PredictionData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    child_first_name = db.Column(db.String(50), nullable=False)
    child_last_name = db.Column(db.String(50), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    sex = db.Column(db.String(1), nullable=False)
    vitamin_a = db.Column(db.String(3), nullable=False)
    birth_order = db.Column(db.Integer, nullable=False)
    breastfeeding = db.Column(db.String(3), nullable=False)
    comorbidity_status = db.Column(db.String(3), nullable=False)
    type_of_birth = db.Column(db.String(50), nullable=False)
    size_at_birth = db.Column(db.String(50), nullable=False)
    dietary_diversity_score = db.Column(db.String(50), nullable=False)
    mothers_age = db.Column(db.Integer, nullable=False)
    residence = db.Column(db.String(50), nullable=False)
    mothers_education_level = db.Column(db.String(50), nullable=False)
    fathers_education_level = db.Column(db.String(50), nullable=False)
    womens_autonomy_tertiles = db.Column(db.String(50), nullable=False)
    toilet_facility = db.Column(db.String(20), nullable=False)
    source_of_drinking_water = db.Column(db.String(50), nullable=False)
    bmi_of_mother = db.Column(db.String(50), nullable=False)
    number_of_children_under_five = db.Column(db.Integer, nullable=False)
    media_exposure = db.Column(db.String(3), nullable=False)
    mothers_working_status = db.Column(db.String(50), nullable=False)
    household_size = db.Column(db.Integer, nullable=False)
    wealth_quantile = db.Column(db.String(50), nullable=False)
    prediction_result = db.Column(db.String(50), nullable=False)
    parent_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    admin_id = db.Column(db.Integer, nullable=False)
    user = db.relationship('User', backref=db.backref('predictions', lazy=True))

    household_id = db.Column(db.Integer, db.ForeignKey('household.id'), nullable=True)
    household = db.relationship('Household', backref=db.backref('predictions', lazy=True))
    
    __table_args__ = (db.UniqueConstraint('child_first_name', 'child_last_name', 'parent_id', name='uq_child_name_parent'),)

    def __repr__(self):
        return f'<PredictionData {self.id}>'



class PredictionForm(FlaskForm):
    child_first_name = StringField("Child's First Name", validators=[DataRequired()])
    child_last_name = StringField("Child's Last Name", validators=[DataRequired()])
    age = IntegerField('Age in months', validators=[DataRequired()])
    sex = SelectField('Sex', choices=[('M', 'Male'), ('F', 'Female')], validators=[DataRequired()])
    vitamin_a = SelectField('Vitamin A', choices=[('Yes', 'Yes'), ('No', 'No')], validators=[DataRequired()])
    birth_order = IntegerField('Birth Order', validators=[DataRequired()])
    breastfeeding = SelectField('Breastfeeding', choices=[('Yes', 'Yes'), ('No', 'No')], validators=[DataRequired()])
    comorbidity_status = SelectField('Comorbidity Status', choices=[('Yes', 'Yes'), ('No', 'No')], validators=[DataRequired()])
    type_of_birth = SelectField('Type of Birth', choices=[('Singleton', 'Singleton'), ('Multiple', 'Multiple')], validators=[DataRequired()])
    size_at_birth = SelectField('Size at Birth', choices=[('Smaller than average', 'Smaller than average'), ('Average', 'Average'), ('Larger than average', 'Larger than average')], validators=[DataRequired()])
    dietary_diversity_score = SelectField('Dietary Diversity Score', choices=[('Below minimum requirement', 'Below minimum requirement'), ('Minimum Requirement', 'Minimum Requirement'), ('Maximum Requirement', 'Maximum Requirement')], validators=[DataRequired()])
    mothers_age = IntegerField("Mother's Age", validators=[DataRequired()])
    residence = SelectField('Residence', choices=[('Rural', 'Rural'), ('Urban', 'Urban')], validators=[DataRequired()])
    mothers_education_level = SelectField("Mother's Education Level", choices=[('Elementary', 'Elementary'), ('Highschool', 'Highschool'), ('College', 'College')], validators=[DataRequired()])
    fathers_education_level = SelectField("Father's Education Level", choices=[('Elementary', 'Elementary'), ('Highschool', 'Highschool'), ('College', 'College')], validators=[DataRequired()])
    womens_autonomy_tertiles = SelectField("Women's Autonomy Tertiles", choices=[('Low', 'Low'), ('Medium', 'Medium'), ('High', 'High')], validators=[DataRequired()])
    toilet_facility = SelectField("Toilet Facility", choices=[('Improved', 'Improved'), ('Unimproved', 'Unimproved')], validators=[DataRequired()])
    source_of_drinking_water = SelectField("Source of Drinking Water", choices=[('Improved', 'Improved'), ('Unimproved', 'Unimproved')], validators=[DataRequired()])
    bmi_of_mother = SelectField("BMI of Mother", choices=[('Underweight', 'Underweight'), ('Normal', 'Normal'), ('Overweight', 'Overweight'), ('Obese', 'Obese')], validators=[DataRequired()])
    number_of_children_under_five = IntegerField("Number of Children Under Five", validators=[DataRequired()])
    media_exposure = SelectField("Media Exposure", choices=[('Yes', 'Yes'), ('No', 'No')], validators=[DataRequired()])
    mothers_working_status = SelectField("Mother's Working Status", choices=[('Working', 'Working'), ('Not Working', 'Not Working')], validators=[DataRequired()])
    household_size = IntegerField("Household Size", validators=[DataRequired()])
    wealth_quantile = SelectField("Wealth Quantile", choices=[('Poor', 'Poor'),('Low Income', 'Low Income'), ('Low Middle Income', 'Low Middle Income'), ('Middle Income', 'Middle Income'), ('Upper Middle Income', 'Upper Middle Income'), ('Upper Income', 'Upper Income'), ('Rich', 'Rich')], validators=[DataRequired()])
    submit = SubmitField('Submit')


class NewUserForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(max=50)])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=6, max=50)])
    barangay = SelectField('Barangay', validators=[DataRequired()], choices=[
        'Aganan', 'Amparo', 'Anilao', 'Balabag', 'Cabugao Norte', 'Cabugao Sur',
        'Jibao-an', 'Mali-ao', 'Pagsanga-an', 'Pal-agon', 'Pandac', 'Purok 1',
        'Purok 2', 'Purok 3', 'Purok 4', 'Tigum', 'Ungka 1', 'Ungka 2'
    ])
    submit = SubmitField('Add User')

    
# # create admin Class
# class Admin(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     username = db.Column(db.String(255), nullable=False)
#     password = db.Column(db.String(255), nullable=False)
#     barangay = db.Column(db.String(80), nullable=False)

#     def __repr__(self):
#         return f'Admin("{self.username}","{self.id}")'

class DummyForm(FlaskForm):
    hidden_tag = HiddenField('hidden_tag')


# Ensure user_accounts is defined globally
# user_accounts = {}

# def load_user_accounts():
#     global user_accounts
#     user_accounts.clear()
#     users = User.query.all()
#     for user in users:
#         if user.barangay not in user_accounts:
#             user_accounts[user.barangay] = []
#         user_accounts[user.barangay].append({
#             'id': user.id,
#             'username': user.username,
#             'email': user.email,
#             'password': user.password,
#             'barangay': user.barangay,
#             'status': user.status
#         })


# Initialize the database
def create_db_tables():
    with app.app_context():
        db.create_all()
        print("Database tables created.")

# Create table
create_db_tables()


# Custom Jinja2 filter to format numbers
def format_number(value):
    if value.is_integer():
        return int(value)
    return round(value, 1)

app.jinja_env.filters['format_number'] = format_number


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Main index page
@app.route('/')
def index():
    return render_template('System4/index.html')

# User Type
@app.route('/user-type-redirect', methods=['POST'])
def userTypeRedirect():
    user_type = request.form.get('user-type')
    
    if user_type == 'guardian':
        return redirect(url_for('userIndex'))
    elif user_type == 'health_worker':
        return redirect(url_for('adminIndex'))  # Assuming you have a separate login route for health workers
    else:
        flash('Invalid user type selected.', 'danger')
        return redirect(url_for('index'))

#-------------------------Admin Area---------------------------------------

# Admin login
@app.route('/admin/', methods=['GET', 'POST'])
def adminIndex():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        admin_id = authenticate_admin(username, password)
        if admin_id:
            session['admin_id'] = admin_id
            return redirect(url_for('adminDashboard'))
        else:
            flash('Invalid username or password')
    return render_template('admin/login.html')

# Route to view all barangays
@app.route('/admin/barangays')
def adminBarangays():
    if 'admin_id' in session:
        # Use a set to ensure unique barangays
        unique_barangays = list(set(barangays))
        unique_barangays.sort()  # Sort the barangays for better readability
        return render_template('admin/barangays.html', barangays=unique_barangays)
    else:
        flash('You are not logged in or do not have the required permissions.')
        return redirect(url_for('adminIndex'))


@app.route('/admin/user_barangays')
def adminUserBarangays():
    if not session.get('admin_id'):
        flash('You are not logged in or do not have the required permissions.', 'danger')
        return redirect(url_for('adminIndex'))

    barangays = ['Aganan', 'Amparo', 'Anilao', 'Balabag', 'Cabugao Norte', 'Cabugao Sur', 'Jibao-an', 'Mali-ao', 'Pagsanga-an', 'Pal-agon',
                 'Pandac', 'Purok 1', 'Purok 2', 'Purok 3', 'Purok 4', 'Tigum', 'Ungka 1', 'Ungka 2']

    return render_template('admin/userbarangay.html', barangays=barangays)


# Admin Dashboard
@app.route('/admin/dashboard')
def adminDashboard():
    if 'admin_id' in session:
        total_users = User.query.count()
        
        low_risk_count = PredictionData.query.filter_by(prediction_result='Low Risk').count()
        medium_risk_count = PredictionData.query.filter_by(prediction_result='Mid Risk').count()
        high_risk_count = PredictionData.query.filter_by(prediction_result='High Risk').count()

        return render_template(
            'admin/home.html', 
            total_users=total_users, 
            low_risk_count=low_risk_count, 
            medium_risk_count=medium_risk_count, 
            high_risk_count=high_risk_count
        )
    else:
        flash('You are not logged in or do not have the required permissions.')
        return redirect(url_for('adminIndex'))


# Admin Get All User
@app.route('/admin/get_all_users')
def adminGetAllUser():
    if 'admin_id' in session:
        users = User.query.all()
        form = DummyForm()  # Create an instance of DummyForm
        return render_template('admin/all-user.html', users=users, form=form)
    else:
        flash('You are not logged in or do not have the required permissions.')
        return redirect(url_for('adminIndex'))

@app.route('/admin/user_profiles/<barangay>')
def adminUserProfiles(barangay):
    if 'admin_id' not in session:
        flash('You are not logged in or do not have the required permissions.', 'danger')
        return redirect(url_for('adminIndex'))
    
    users = User.query.filter_by(barangay=barangay).all()
    return render_template('admin/user_profiles.html', users=users, barangay=barangay)


# Admin View Prediction Data
@app.route('/admin/predictions')
def adminPredictionData():
    if 'admin_id' in session:
        predictions = PredictionData.query.all()
        users = {user.id: user for user in User.query.all()}
        
        prediction_list = []
        for prediction in predictions:
            user = users.get(prediction.parent_id)
            prediction_list.append({
                'id': prediction.id,
                'child_first_name': prediction.child_first_name,
                'child_last_name': prediction.child_last_name,
                'age': prediction.age,
                'sex': prediction.sex,
                'prediction_result': prediction.prediction_result,
                'household_id': prediction.household_id  # Assuming this field exists
            })
        
        return render_template('admin/viewpredictions.html', predictions=prediction_list)
    else:
        flash('You are not logged in or do not have the required permissions.')
        return redirect(url_for('adminIndex'))

# Route to view predictions filtered by barangay
@app.route('/admin/predictions/<string:barangay>', methods=['GET'])
def adminBarangayPredictions(barangay):
    if 'admin_id' not in session:
        flash('You are not logged in or do not have the required permissions.')
        return redirect(url_for('adminIndex'))

    households = Household.query.filter_by(address=barangay).all()
    predictions = []
    for household in households:
        household_predictions = PredictionData.query.filter_by(household_id=household.id).all()
        for prediction in household_predictions:
            predictions.append({
                'id': prediction.id,
                'child_first_name': prediction.child_first_name,
                'child_last_name': prediction.child_last_name,
                'age': prediction.age,
                'sex': prediction.sex,
                'prediction_result': prediction.prediction_result,
                'household_address': household.address,
                'mother_first_name': household.mother_first_name,
                'mother_last_name': household.mother_last_name
            })
    return render_template('admin/barangay_predictions.html', predictions=predictions, barangay=barangay)



@app.route('/admin/view_prediction/<int:prediction_id>')
def viewPrediction(prediction_id):
    if 'admin_id' in session:
        prediction = PredictionData.query.get_or_404(prediction_id)
        user = User.query.get(prediction.parent_id)
        
        prediction_details = {
            'id': prediction.id,
            'child_first_name': prediction.child_first_name,
            'child_last_name': prediction.child_last_name,
            'age': prediction.age,
            'sex': prediction.sex,
            'prediction_result': prediction.prediction_result,
            'user_full_name': f"{user.fname} {user.lname}"
        }

        # Fetch the household data related to the prediction's admin
        household = Household.query.filter_by(admin_id=prediction.parent_id).first()

        # Store prediction and household data in session
        session['prediction'] = prediction.__dict__
        session['household'] = household.__dict__ if household else None

        return redirect(url_for('adminViewResults'))
    else:
        flash('You are not logged in or do not have the required permissions.')
        return redirect(url_for('adminIndex'))

# Admin Approve
@app.route('/admin/approve/<int:id>', methods=['POST'])
def adminApprove(id):
    user = User.query.get_or_404(id)
    user.status = 1
    db.session.commit()
    flash('User has been approved.')
    return redirect(url_for('adminGetAllUser'))

# Admin Reject
@app.route('/admin/reject/<int:id>', methods=['POST'])
def adminReject(id):
    user = User.query.get_or_404(id)
    user.status = 2
    db.session.commit()
    flash('User has been rejected.')
    return redirect(url_for('adminGetAllUser'))

@app.route('/admin/add_user', methods=['GET', 'POST'])
def adminAddUser():
    if not session.get('admin_id'):
        flash('You are not logged in or do not have the required permissions.', 'danger')
        return redirect(url_for('adminIndex'))

    form = NewUserForm()
    
    if form.validate_on_submit():
        new_user = User(
            username=form.username.data,
            barangay=form.barangay.data
        )
        new_user.set_password(form.password.data)  # Hash the password before saving

        # Add new user to the database
        try:
            db.session.add(new_user)
            db.session.commit()
            flash('New user created successfully!', 'success')
            return redirect(url_for('adminDashboard'))
        except IntegrityError:
            db.session.rollback()
            flash('Username already exists. Please choose a different username.', 'danger')
    
    return render_template('admin/admin_add_user.html', form=form)

@app.route('/admin/edit_user/<int:user_id>', methods=['GET', 'POST'])
def adminEditUser(user_id):
    if not session.get('admin_id'):
        flash('You are not logged in or do not have the required permissions.', 'danger')
        return redirect(url_for('adminIndex'))

    user = User.query.get_or_404(user_id)
    form = NewUserForm(obj=user)

    if form.validate_on_submit():
        user.username = form.username.data
        user.barangay = form.barangay.data
        if form.password.data:
            user.set_password(form.password.data)

        try:
            db.session.commit()
            flash('User profile updated successfully!', 'success')
            return redirect(url_for('adminUserProfiles', barangay=user.barangay))
        except IntegrityError:
            db.session.rollback()
            flash('Username already exists. Please choose a different username.', 'danger')

    return render_template('admin/admin_edit_user.html', form=form, user=user)

@app.route('/admin/delete_user/<int:user_id>', methods=['GET', 'POST'])
def adminDeleteUser(user_id):
    if not session.get('admin_id'):
        flash('You are not logged in or do not have the required permissions.', 'danger')
        return redirect(url_for('adminIndex'))

    user = User.query.get(user_id)
    if not user:
        flash('User not found', 'danger')
        return redirect(url_for('adminUserBarangays'))

    try:
        # Delete the associated households first if they exist
        households = Household.query.filter_by(user_id=user_id).all()
        for household in households:
            db.session.delete(household)
        
        db.session.delete(user)
        db.session.commit()
        flash('User deleted successfully', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Error deleting user: {e}', 'danger')

    return redirect(url_for('adminUserBarangays'))

# Admin logout
@app.route('/admin/logout')
def adminLogout():
    session.clear()
    return redirect(url_for('adminIndex'))

#--------------------------------user content--------------------------------
# User login
@app.route('/user/', methods=['GET', 'POST'])
def userIndex():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = User.query.filter_by(username=username).first()

        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            flash('Login successful!', 'success')
            return redirect(url_for('userDashboard'))
        else:
            flash('Invalid username or password.', 'danger')
            return redirect(url_for('userIndex'))

    return render_template('System4/login.html')


# User dashboard
@app.route('/user/dashboard')
def userDashboard():
    if not session.get('user_id'):
        return redirect('/user/')
    
    user_id = session.get('user_id')
    user = User.query.filter_by(id=user_id).first()

    # Fetch household related to the user
    household = Household.query.filter_by(user_id=user_id).first()

    # Fetch predictions related to the user
    predictions = PredictionData.query.filter_by(parent_id=user_id).all()

    return render_template('System4/home.html', title="User Dashboard", user=user, household=household, predictions=predictions)


@app.route('/user/results/<int:prediction_id>')
def userViewResults(prediction_id):
    if 'user_id' not in session:
        flash('You are not logged in.')
        return redirect(url_for('userIndex'))

    user_id = session.get('user_id')
    prediction = PredictionData.query.filter_by(id=prediction_id, parent_id=user_id).first()

    if not prediction:
        flash('Prediction not found or you do not have access to this prediction.')
        return redirect(url_for('userDashboard'))

    household = Household.query.get(prediction.household_id)

    return render_template('System4/results.html', prediction=prediction, household=household)

@app.route('/user/view_profile', methods=['GET', 'POST'])
def viewProfile():
    user_id = session.get('user_id')

    if not user_id:
        flash('You need to be logged in to view this page.', 'danger')
        return redirect(url_for('userIndex'))  # Redirect to the login page if not logged in

    form = HouseholdForm()

    # Fetch household data from the database
    household = Household.query.filter_by(user_id=user_id).first()
    user = User.query.get(user_id)

    if not user:
        flash('User not found.', 'danger')
        return redirect(url_for('userDashboard'))  # Redirect to user's dashboard or any relevant page

    if household:
        form.address.data = household.address
        form.mother_first_name.data = household.mother_first_name
        form.mother_last_name.data = household.mother_last_name
        form.father_first_name.data = household.father_first_name
        form.father_last_name.data = household.father_last_name
        form.mother_age.data = household.mother_age
        form.father_age.data = household.father_age
    else:
        # Use the address from the user if no household information is present
        form.address.data = user.barangay

    if form.validate_on_submit():
        if household:
            household.address = form.address.data
            household.mother_first_name = form.mother_first_name.data
            household.mother_last_name = form.mother_last_name.data
            household.father_first_name = form.father_first_name.data
            household.father_last_name = form.father_last_name.data
            household.mother_age = form.mother_age.data
            household.father_age = form.father_age.data
        else:
            household = Household(
                user_id=user_id,
                address=form.address.data,
                mother_first_name=form.mother_first_name.data,
                mother_last_name=form.mother_last_name.data,
                father_first_name=form.father_first_name.data,
                father_last_name=form.father_last_name.data,
                mother_age=form.mother_age.data,
                father_age=form.father_age.data
            )
            db.session.add(household)
        db.session.commit()
        flash('Profile information updated successfully.', 'success')
        return redirect(url_for('viewProfile'))  # Redirect to the same page after saving

    return render_template('System4/view_profile.html', form=form, user=user)


# User Logout
@app.route('/user/logout')
def userLogout():
    if not session.get('user_id'):
        return redirect('/user/')
    
    session.clear()  # Clear all session data
    flash('You have been logged out', 'success')
    return redirect('/user/')

# User Change Password
@app.route('/user/change-password', methods=['POST', 'GET'])
def userChangePassword():
    if not session.get('user_id'):
        return redirect('/user/')
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        if email == "" or password == "":
            flash('Please fill the field.', 'warning')
            return redirect('/user/change-password')
        else:
            # Find user by email
            user = User.query.filter_by(email=email).first()
            if user:
                # Update password as plain text
                user.password = password
                db.session.commit()
                flash('Password changed successfully.', 'success')
                return redirect('/user/change-password')
            else:
                flash('Invalid email', 'danger')
                return redirect('/user/change-password')
    else:
        return render_template('user/change-password.html', title="Change Password")

# About
@app.route('/user/about')
def about():
    if not session.get('user_id'):
        flash('User not logged in', 'danger')
        return redirect(url_for('userIndex'))
    return render_template('System4/about.html', title="e-OPT")

# --------------------------- Child Section -------------------------------------
    
# Child Information Router    
@app.route('/user/add_child', methods=['GET', 'POST'])
def add_child():
    if not session.get('user_id'):
        flash('User not logged in', 'danger')
        return redirect(url_for('userIndex'))
    
    form = ChildForm()
    if form.validate_on_submit():
        new_child = Child(
            first_name=form.first_name.data,
            last_name=form.last_name.data,
            user_id=session.get('user_id'),
            parent_id=session.get('user_id')
        )
        db.session.add(new_child)
        db.session.commit()
        flash('Child information added successfully', 'success')
        return redirect(url_for('userDashboard'))

    return render_template('System4/precord.html', form=form)

# Add Household information
@app.route('/admin/add_household', methods=['GET', 'POST'])
def adminAddHousehold():
    user_id = request.args.get('user_id')
    form = HouseholdForm()

    # Fetch household data from the database
    household = Household.query.filter_by(user_id=user_id).first()
    user = User.query.get(user_id)
    if not user:
        flash('User not found.', 'danger')
        return redirect(url_for('adminUserBarangays'))

    if household:
        form.address.data = household.address
        form.mother_first_name.data = household.mother_first_name
        form.mother_last_name.data = household.mother_last_name
        form.father_first_name.data = household.father_first_name
        form.father_last_name.data = household.father_last_name
        form.mother_age.data = household.mother_age
        form.father_age.data = household.father_age
    else:
        # Use the address from the user if no household information is present
        form.address.data = user.barangay

    if form.validate_on_submit():
        if household:
            household.address = form.address.data
            household.mother_first_name = form.mother_first_name.data
            household.mother_last_name = form.mother_last_name.data
            household.father_first_name = form.father_first_name.data
            household.father_last_name = form.father_last_name.data
            household.mother_age = form.mother_age.data
            household.father_age = form.father_age.data
        else:
            household = Household(
                user_id=user_id,
                address=form.address.data,
                mother_first_name=form.mother_first_name.data,
                mother_last_name=form.mother_last_name.data,
                father_first_name=form.father_first_name.data,
                father_last_name=form.father_last_name.data,
                mother_age=form.mother_age.data,
                father_age=form.father_age.data
            )
            db.session.add(household)
        db.session.commit()
        flash('Household information saved successfully.', 'success')
        return redirect(url_for('adminUserBarangays'))
    
    return render_template('admin/add_household.html', form=form, user=user)





# ---------------------------- Prediction Section -------------------------------------------------- #
@app.route('/admin/predict', methods=['GET', 'POST'])
def adminPredict():
    if not session.get('admin_id'):
        flash('You are not logged in or do not have the required permissions.', 'danger')
        return redirect(url_for('adminIndex'))

    user_id = request.args.get('user_id')
    form = PredictionForm()

    household = Household.query.filter_by(user_id=user_id).first()
    if household:
        form.child_last_name.data = household.father_last_name
        form.mothers_age.data = household.mother_age

    if form.validate_on_submit():
        try:
            admin_id = session.get('admin_id')
            user = User.query.get(user_id)
            household = Household.query.filter_by(user_id=user_id).first()

            # Fetch form data
            child_first_name = form.child_first_name.data
            child_last_name = form.child_last_name.data

            # Check for existing prediction for the child under the same parent
            existing_prediction = PredictionData.query.filter_by(
                child_first_name=child_first_name,
                child_last_name=child_last_name,
                parent_id=user_id
            ).first()

            if existing_prediction:
                flash('Prediction for this child already exists.', 'warning')
                return redirect(url_for('adminPredict', user_id=user_id))

            # Continue fetching the rest of the form data
            age = form.age.data
            sex = form.sex.data
            vitamin_a = form.vitamin_a.data
            birth_order = form.birth_order.data
            breastfeeding = form.breastfeeding.data
            comorbidity_status = form.comorbidity_status.data
            type_of_birth = form.type_of_birth.data
            size_at_birth = form.size_at_birth.data
            dietary_diversity_score = form.dietary_diversity_score.data
            mothers_age = form.mothers_age.data
            residence = form.residence.data
            mothers_education_level = form.mothers_education_level.data
            fathers_education_level = form.fathers_education_level.data
            womens_autonomy_tertiles = form.womens_autonomy_tertiles.data
            toilet_facility = form.toilet_facility.data
            source_of_drinking_water = form.source_of_drinking_water.data
            bmi_of_mother = form.bmi_of_mother.data
            number_of_children_under_five = form.number_of_children_under_five.data
            media_exposure = form.media_exposure.data
            mothers_working_status = form.mothers_working_status.data
            household_size = form.household_size.data
            wealth_quantile = form.wealth_quantile.data

            # Map form data for prediction
            bmi_mapping = {'Underweight': 0, 'Normal': 1, 'Overweight': 2, 'Obese': 3}
            wealth_quantile_mapping = {
                'Poor': 0,
                'Low Income': 1,
                'Low Middle Income': 2,
                'Middle Income': 3,
                'Upper Middle Income': 4,
                'Upper Income': 5,
                'Rich': 6
            }

            data = np.array([[age, 1 if sex == 'M' else 0, 1 if vitamin_a == 'Yes' else 0, birth_order,
                              1 if breastfeeding == 'Yes' else 0, 1 if comorbidity_status == 'Yes' else 0,
                              1 if type_of_birth == 'Singleton' else 0,
                              {'Smaller than average': 0, 'Average': 1, 'Larger than average': 2}[size_at_birth],
                              {'Below minimum requirement': 0, 'Minimum Requirement': 1, 'Maximum Requirement': 2}[dietary_diversity_score],
                              mothers_age, 1 if residence == 'Urban' else 0,
                              {'Elementary': 0, 'Highschool': 1, 'College': 2}[mothers_education_level],
                              {'Elementary': 0, 'Highschool': 1, 'College': 2}[fathers_education_level],
                              {'Low': 0, 'Medium': 1, 'High': 2}[womens_autonomy_tertiles], 1 if toilet_facility == 'Improved' else 0,
                              1 if source_of_drinking_water == 'Improved' else 0, bmi_mapping[bmi_of_mother], number_of_children_under_five,
                              1 if media_exposure == 'Yes' else 0, 1 if mothers_working_status == 'Working' else 0, household_size,
                              wealth_quantile_mapping[wealth_quantile]]])

            prediction_result = model.predict(data)[0]
            result = 'Low Risk' if prediction_result == 0 else 'Mid Risk' if prediction_result == 1 else 'High Risk'

            # Create new prediction
            new_prediction = PredictionData(
                child_first_name=child_first_name,
                child_last_name=child_last_name,
                age=age, sex=sex, vitamin_a=vitamin_a, birth_order=birth_order, breastfeeding=breastfeeding,
                comorbidity_status=comorbidity_status, type_of_birth=type_of_birth, size_at_birth=size_at_birth,
                dietary_diversity_score=dietary_diversity_score, mothers_age=mothers_age, residence=residence,
                mothers_education_level=mothers_education_level, fathers_education_level=fathers_education_level,
                womens_autonomy_tertiles=womens_autonomy_tertiles, toilet_facility=toilet_facility,
                source_of_drinking_water=source_of_drinking_water, bmi_of_mother=bmi_of_mother,
                number_of_children_under_five=number_of_children_under_five, media_exposure=media_exposure,
                mothers_working_status=mothers_working_status, household_size=household_size,
                wealth_quantile=wealth_quantile, prediction_result=result, parent_id=user_id,
                household_id=household.id, admin_id=admin_id
            )

            # Add to database
            db.session.add(new_prediction)
            db.session.commit()

            # Redirect to results page
            flash(f'Prediction: {result}', 'success')
            return redirect(url_for('adminResults', prediction_id=new_prediction.id))

        except IntegrityError:
            db.session.rollback()
            flash('Prediction for this child already exists.', 'warning')
            return redirect(url_for('adminPredict', user_id=user_id))
        except Exception as e:
            print(f"Error: {e}")
            flash(f'Error making prediction: {e}', 'danger')
    else:
        flash('Form validation failed. Please check the inputs.', 'danger')

    return render_template('admin/predict.html', form=form, household=household, user_id=user_id)



@app.route('/admin/results/<int:prediction_id>')
def adminResults(prediction_id):
    if not session.get('admin_id'):
        flash('You are not logged in or do not have the required permissions.', 'danger')
        return redirect(url_for('adminIndex'))

    prediction = PredictionData.query.filter_by(id=prediction_id).first()
    if not prediction:
        flash('Prediction not found', 'danger')
        return redirect(url_for('adminDashboard'))

    household = Household.query.filter_by(id=prediction.household_id).first()

    return render_template('admin/results.html', prediction=prediction, household=household)

@app.route('/admin/viewResults/<int:prediction_id>')
def adminViewResultsButton(prediction_id):
    prediction = PredictionData.query.get(prediction_id)
    household = Household.query.get(prediction.household_id) if prediction else None

    if not prediction:
        return redirect(url_for('adminDashboard'))
    
    return render_template('admin/results.html', prediction=prediction, household=household)

# Delete Prediction
@app.route('/admin/delete_prediction/<int:prediction_id>', methods=['POST'])
def deletePrediction(prediction_id):
    if 'admin_id' in session:
        prediction = PredictionData.query.get_or_404(prediction_id)
        try:
            barangay = prediction.household.address  # Get the barangay from the household before deletion
            db.session.delete(prediction)
            db.session.commit()
            flash('Prediction deleted successfully.', 'success')
        except Exception as e:
            db.session.rollback()
            flash(f'Error deleting prediction: {e}', 'danger')
        # Redirect back to the barangay predictions page after deletion
        return redirect(url_for('adminBarangayPredictions', barangay=barangay))
    else:
        flash('You are not logged in or do not have the required permissions.', 'danger')
        return redirect(url_for('adminIndex'))

if __name__ == '__main__': 
    app.run(debug=True)