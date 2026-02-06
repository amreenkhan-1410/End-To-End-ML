import streamlit as st
import pandas as pd
import numpy as np
import os 
import seaborn as sns
import matplotlib.pyplot as plt 
import requests
from datetime import datetime
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.metrics import accuracy_score,confusion_matrix

# logger
def log(message):
    timestamp =datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}]{message}")

#session state
if "clean_saved" not in st.session_state:
    st.session_state.clean_saved = False

#folder setup
BASE_DIR=os.path.dirname(os.path.abspath(__file__))
RAW_DIR=os.path.join(BASE_DIR,"data","raw")
CLEAN_DIR=os.path.join(BASE_DIR,"data","cleaned")

os.makedirs(RAW_DIR,exist_ok =True)
os.makedirs(CLEAN_DIR,exist_ok=True)

log("Application started")
log(f"RAW_DIR={RAW_DIR}")
log(f"CLEAN_DIR={CLEAN_DIR}")

# page config

st.set_page_config("END-TO-END SVM Platform", layout="wide")
st.title("END-TO-END SVM Platform")

# sidebar : Model settings

st.sidebar.header("SVM settings")

Kernel=st.sidebar.selectbox("Kernel",["linear","rbf","poly","sigmoid"])


C = st.sidebar.slider("C (Regularization)",0.01,10.0,1.0)
gamma = st.sidebar.selectbox("Gamma",["scale","auto"])

log(f"svm settings --->kernel = {Kernel},C={C},Gamma={gamma}")

# Step-1 :Data Ingestion
st.header("Step:1 Data Ingestion")
log("Step 1 started : Data Ingestion")

option = st.radio("Choose data Source",["Download Dataset","Upload CSV"])

df = None
raw_path = None

if option == "Download Dataset":
    if st.button("Download Iris Dataset"):
        log("Downloading Iris Dataset")
        url= "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
        response = requests.get(url)


        raw_path = os.path.join(RAW_DIR,"iris.csv")
        with open(raw_path,"wb") as f:
            f.write(response.content)

        df=pd.read_csv(raw_path)
        st.success("Dataset Downloaded Successfully")
        log(f"iris dataset saved at {raw_path}")

if option == "Upload CSV":
    uploaded_file=st.file_uploader("upload csv file", type=["csv"])
    if uploaded_file:
        raw_path=os.path.join(RAW_DIR,uploaded_file.name)
        with open(raw_path,"wb") as f:
            f.write(uploaded_file.getbuffer())
        df=pd.read_csv(raw_path)
        st.success("File Uploaded Successfully ")
        log(f"uploaded dataset saved at {raw_path}")

# Step 2 : EDA 

if df is not None:
    st.header("Step2:Exploratory Data Analysis")
    log("Step 2 started EDA")

    st.dataframe(df.head())
    st.write("Shape:",df.shape)
    st.write("Missig values",df.isnull().sum())

    fig,ax = plt.subplots()
    sns.heatmap(df.corr(numeric_only=True),annot=True,cmap = "coolwarm" , ax=ax)
    st.pyplot(fig)

    log("EDA completed")

#step 3: Data cleaning
if df is not None:
    st.header("Step3 : Data Cleaning")

    strategy = st.selectbox(
        "Missing value strategy",
        ["Mean","Median","Drop Rows"]
    )

    df_clean =df.copy()

    if strategy == "Drop Rows":
        df_clean = df_clean.dropna()
    
    else:
        for col in df_clean.select_dtypes(include = np.number):
            if strategy=="Mean":
                df_clean[col]= df_clean[col].fillna(df_clean[col].mean())
            else:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            
    st.session_state.df_clean = df_clean
    st.success("Data cleaning completed")

else:
    st.info("Please complete the step 1 ( Data Ingestion) First..")


# step 4: save the cleaned data 

if st.button("Save Cleaned Dataset"):
    if st.session_state.df_clean is None:
        st.error("No cleaned Data Found")
    else:
        timestamp =datetime.now().strftime("%Y%m%d_%H%M%S")
        clean_filename=f"cleaned_dataset_{timestamp}.csv"
        clean_path=os.path.join(CLEAN_DIR,clean_filename)

        st.session_state.df_clean.to_csv(clean_path,index= False)

        st.success("Cleaned data saved")
        st.info(f"saved at :{clean_path}")

        log(f"cleaned data saved at {clean_path}")

#step 5: Load cleaned data

st.header("step 5: Load cleaned dataset")
clean_files=os.listdir(CLEAN_DIR)

if not clean_files:
    st.warning("No cleaned Data Found")
    log("No cleaning Datasets found.")

else:
    selected=st.selectbox("select cleaned dataset",clean_files)
    df_model=pd.read_csv(os.path.join(CLEAN_DIR,selected))

    st.success(f"Loaded dataset:{selected}")
    log((f"Loaded  cleaned dataset:{selected}"))
    
    st.dataframe(df_model.head())

#step 6 : train svm 

st.header("Step 6: Train SVM")
log("Step 6 started svm training ")

target = st.selectbox("Select target column ",df_model.columns)

y= df_model[target]


#validate target for the classification

if y.dtype != "object" and y.nunique()>20 :
    st.error("Invalid target selection."
             "SVM classifier requires CATEGORICAL LABELS."
             "Please select the categorical column(e.g..'species')"
            )
    st.stop()

#encode target if categorical

if y.dtype == "object":
    y=LabelEncoder().fit_transform(y)
    log("Target column encoded")

#select nummeric features only

x=df_model.drop(columns=[target])
x=x.select_dtypes(include=np.number)

if x.empty:
    st.error("No Numerical features available for training")
    st.stop()

# scale features 

scaler = StandardScaler()
x=scaler.fit_transform(x)

#Train test split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=42)

#model selection
model=SVC(kernel =Kernel,C=C,gamma =gamma)
model.fit(x_train,y_train)

#Evaluate

y_pred =model.predict(x_test)

acc= accuracy_score(y_test,y_pred)

st.success(f"Accuracy:{acc}")
log(f"svm trained successfully | Accuracy = {acc:.2f}")

cm=confusion_matrix(y_test,y_pred)
fig,ax =plt.subplots()
sns.heatmap(cm,annot = True,fmt = "d",cmap="Blues",ax = ax)
st.pyplot(fig)
