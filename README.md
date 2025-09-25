**Study's Objective and Structure:**

Our study, “Preventive Maintenance for Marine Engines”, focuses on building a predictive model to detect early signs of technical failure in marine engines, 
aiming to support timely maintenance and prevent accidents or delays in the supply chain. Our approach begins with exploring the data to understand the structure 
of the data and to identify potentially relevant features. Subsequently, unsupervised learning techniques are used to uncover patterns and groupings in the sensor and 
operational data, aiding in feature selection and anomaly detection. Finally, supervised machine learning models are developed to predict maintenance needs. 
A multinomial logistic regression model is serving as the baseline, and its performance is then compared against more complex algorithms such as Random Forest 
and Support Vector Machines (SVM). The central research question guiding this project is the following: 

How can machine learning models predict early signs of maintenance needs in marine engines using engine performance metrics, operational conditions, and failure modes data?

**File Structure**

In the following repository you will find:

  - marine_engine_data.csv : weekly time series of simulated data collected during the operation of 50 marine engines totalizing 5200 instances (data use during the entire project)
  - DataExplo.qmd : Exploratory Data Analysis of our dataset
  - UnsupLearning_clust.R : Unsupervised Clustering analysis of the main features
  - UnsupLearning.R : Principal Component Analysis after cluster creation
  - SupervisedLearning_LogReg.R : Multinomial Logistic Regression which is the baseline of our modeling section
  - SupervisedLearning_RF.R : Random Forest modeling
  - SupervisedLearning_SVM.R : Support Vector Machine modeling
  - models_comparison.R : Comparison between the 3 models that we have constructed
  - Group_D.pdf : Final study report
  - Group_D_presentation_slides.pdf : Presentation report of the study
