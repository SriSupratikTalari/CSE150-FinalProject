# CSE150aProject Final Submission


## 1. Describe your Data

- [Link to Data Set](https://www.kaggle.com/datasets/meharshanali/amazon-stocks-2025)
- **Primary Features For Our Use:**
  -High (Highest price of Amazon for that day)
  -Close (Closing price of Amazon for that day)
- **Task:**
  -The purpose of using this dataset is trying to find out how many times the Amazon Stock price was greater than its overall average price in its entire history.
- **Relevance:**
  - This data is relevent to probabilistic modeling because we are using it to create a Markov Model and performing Viterbi algorithm to find the most likely sequence of our Hidden State.
- **Preprocessing:**
  - Filter our dataset to only only include the High Price and Closing Price
  - One Hot encode all variables to make easy predictions
- **Dataset Info:**
  - The original dataset has 8 features and 6987 observations
  - Before precrocessing the majority of our features are continuous however after one hot encoding they have become categorical
  - Origin: Kaggle
  - The dataset is indeed reliable as it uses accuracte information from Yahoo Finance
## 2. PEAS / Agent Analysis (5 pts)

- **Performance measure:**
  -   Accuracy: How accurate our model is in perdicting whether our Hidden state is a 0 or 1 (below the overall average for High price or lower than the overall average for High price ).
- **Environment:**  
  - Stock Market, Financial Firm
- **Actuators:**  
  - Screen Display
- **Sensors:**  
  - Keyboard (Entry of symptoms, findings, patient answers), CSV file reader
### Problem to solve
The main problem that we are trying to tackle is to get a sense of how volitile Amazon has been since its begining, 
**Background:**  

- Probablistic modeling is appropriate for this problem because accurate modeling can allow us to not
  only more easily diagnose Heart Disease but also start the patient on more preventative measures early on
  if they start showing sympotms that are synonymous with an accurate Heart Disease diagnosis
---

## 2. Agent Setup, Data Preprocessing, Training Setup (10 pts)

### Dataset Exploration

[Give an overview of your dataset, highlighting important variables]

- [Link to Data Set](https://huggingface.co/datasets/muhrafli/heart-diseases/blob/main/heart%20(3).csv)
- Our Data Set has 12 columns and 918 rows. The twelve columns being age, sex, ChestPainType, RestingBP,
  Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope, and HeartDisease. Of these
  twelve, the most important is of course or HeartDisease column since that is what were evaluating the accuracy
  of our model on and Age, Sex, RestingBP, Cholesterol, RestingECG, and ExerciseAngina since these relate
  directly to various factors that can influence heart disease, the other remaining columns though still are
  important for a more comprehensive diagnosis.

### Variable Descriptions

- Age: A numerical value indicating the patients age. Originally the data set has patients ranging from 28-77 but after preprocessing we set any age greater or equal to 50 as 1 and any age less than 50 as 0 for easier training. Since Age isn't a direct symptom of HeartDisease we have edges coming from age to RestingBP, Cholesterol, MaxHR, and FastingBS which each directly have impact on HeartDisease and have corresponding edges to them
- Sex: A value M or F indicating the patients sex. During preprocessing we convert M to 1 and F to 0. Similar to Age, since Sex isn't a direct symptom of HeartDisease we have edges coming from age to RestingBP, Cholesterol, MaxHR, and FastingBS which each directly have impact on HeartDisease and have corresponding edges to them
- ChestPainType: There are 4 possible values for ChestPainType that are evaluated; 'ATA', 'NAP', 'ASY' and TA. ATA corresponds to Atypical Aniga, NAP for Non-Anginal Pain, ASY for asymptomatic or no chestpain and TA for Typical Angina.
- RestingBP: A numerical value indicating the patients resting blood pressure from 0 to 200. After preprocessing, any blood pressure greater than or equal to 129 is converted to a 1 and anything less is a 0.
- Cholesterol: A numerical value indicating the patients cholesterol levels ranging from 0 to 603. After preprocessing, any cholesterol greater than or equal to 200 is converted to 1 and anything less than 200 is a 0.
- FastingBS: A numerical value indicating the patients fasting blood sugar levels either 0 or 1 and no processing was needed as such. 
- RestingECG: There are 3 possible values for RestingECG; 'Normal', 'ST' and 'LVH'. A Normal ECG referse to a healthy heart rhythm, ST referes to where there is frequent changes in the ST segment of an ECG and LVH referes to left ventricular hyperthorphy associated with hypertension. After preprocessing, 'ST' and 'LVH' were converted to 1 and Normal to 0.
- MaxHR: A numerical value associated with the patients max heart rate ranging from 60-202. After preprocessing, a MaxHR of 150 or greater is converted to 1 and anything less 0.
- ExerciseAngina: Exercise Anigna refers to chest pain that occurs during physical activity. It takes a value either Y or N (Yes or No) indicating yes that the patient has chest pain during physical activity or no that the patient does not have chest pain during physical activity. During preprocessing Y is converted to 1 and N to 0
- Oldpeak: Oldpeak is a numerical value that represents ST depression induced in exercise over rest ranges from -2.6 to 6.2 in the dataset. During preprocessing any value greater than 1 is converted to 1 and anything less than 1 is 0.
- ST_Slope: ST slope refers to slope of the ST segment and takes in 3 values; 'Up', 'Flat', and 'Down'. During preprocessing 'Up' and 'Down' get converted to 1 and 'Flat' gets converted to 0. 
- HeartDisease: A numerical value either 0 or 1 indicating if the patient has heart disease or not. No preprocessing was needed. Our most important variable here because this serves as our baseline for evaluating the accuracy of our model.

### Variable Interactions and Model Structure

![Figure Image](MODEL1.png)

- This Bayesian Network model sturcture was chosen in part because of how these variable act upon each other in real life. ST_SLOPE, Oldpeak, ExerciseAngina, RestingECG all act upon HeartDisease and HeartDisease inversely influences them so that explains those edges. HeartDisease can directly influence ChestPainType moreover. Age and Sex directly influence ones Resting BP but RestingBP doesn't have any hold on Age nor Sex just but does have influence on HeartDisease and vice versa. Similar can be said with Cholesterol, MaxHR, and FastingBS.

### Parameter Calculation

-  We look at the values inside the dataframe and found all the values that correlate with our given and calculated our predictions as such
### Library Usage

- Networkx: Networkx was utilized to visualize our graph

Aric A. Hagberg, Daniel A. Schult and Pieter J. Swart, “Exploring network structure, dynamics, and function using NetworkX”, in Proceedings of the 7th Python in Science Conference (SciPy2008), Gäel Varoquaux, Travis Vaught, and Jarrod Millman (Eds), (Pasadena, CA USA), pp. 11–15, Aug 2008

- Sklearn.metrics: Sklearn.metrics was utilized to evaluate our model on various
  accuracy measures

API design for machine learning software: experiences from the scikit-learn project, Buitinck et al., 2013.

---

## 3. Training the Model (5 pts)

### Code Snippet or Link

```python
from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

print(f"Train set size: {train_df.shape[0]} samples")
print(f"Test set size: {test_df.shape[0]} samples")
def our_model(df,given):
    filtered_b = df.copy()
    for col, val in given.items():
        filtered_b = filtered_b[filtered_b[col] == val]
    denom=filtered_b.shape[0]
    num= filtered_b[filtered_b['HeartDisease'] == 1].shape[0]
    prob=num/denom
    prediction=0
    if prob>0.69:
        prediction=1
    else:
        prediction=0
    return prediction
def predictions(df,cols):
    pred=[]
    for i in range(len(df)):
        given={j:df.iloc[i][j] for j in cols}
        pred.append(our_model(df,given))
    return pred
our_pred_train=predictions(train_df,['Age','Sex','RestingBP'])
actual_train=list(train_df['HeartDisease'])
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
print("Accuracy:", accuracy_score(actual_train, our_pred_train))
print("Precision:", precision_score(actual_train, our_pred_train))
print("Recall:", recall_score(actual_train, our_pred_train))
print("F1 Score:", f1_score(actual_train, our_pred_train))
print("Confusion Matrix:\n", confusion_matrix(actual_train, our_pred_train))
our_pred_test=predictions(test_df,['Age','Sex','RestingBP'])
actual_test=list(test_df['HeartDisease'])
print("Accuracy:", accuracy_score(actual_test, our_pred_test))
print("Precision:", precision_score(actual_test, our_pred_test))
print("Recall:", recall_score(actual_test, our_pred_test))
print("F1 Score:", f1_score(actual_test, our_pred_test))
print("Confusion Matrix:\n", confusion_matrix(actual_test, our_pred_test))
```

---

## 4. Conclusion / Results (15 pts)

### Results
Training Results: 
- **Accuracy**: 0.6239782016348774  
- **Precision**: 0.7289377289377289  
- **Recall**: 0.49625935162094764  
- **F1 Score**: 0.5905044510385756  

### Confusion Matrix

|               | Predicted: No | Predicted: Yes |
|---------------|---------------|----------------|
| Actual: No    | 259            | 74             |
| Actual: Yes   | 202            | 199             |


Test Results:
- **Accuracy**: 0.5380434782608695  
- **Precision**: 0.7291666666666666  
- **Recall**: 0.32710280373831774  
- **F1 Score**: 0.45161290322580644  

### Confusion Matrix

|               | Predicted: No | Predicted: Yes |
|---------------|---------------|----------------|
| Actual: No    | 64            | 13             |
| Actual: Yes   | 72            | 35             |
### Interpretation

Looking at the training results above we see that we were able to achieve a higher precision than our recall which means our model is more likely to predict the right outcome when 
someone does indeed have a heart disease. We also see that we have a lot of false negatives which mean that there are a lot people who have heart disease but aren't being flagged by our model. Looking at the test results we see that our accuracy was about 54% which means the model did slightly better then guessing at random when it came to our test dataset. we can also see that when the model predicts heart disease it predicts correctly about 73% of the time which is fairly good.  

### Proposed Improvements

The first thing that we could imporve on is perform hyper parameter tuning on all columns that are ancestors of the heart disease column which might lead to better accuary and overall a better model that improves in all performance measures. Another thing that we can do is do more research on which columns affect heart disease and which don't so that way we can build a model that mainly focues on the key causes of heart diease. Another thing that we can do is do more research on different probablistic models that might work for our dataset. 

---

## 5. Additional Notes
imported libraries: sklearn.metrics
usage: used it to provide functions to evaluate our model 

### Citations

dbell3. (n.d.). DBELL3/CSE-150-proj-1: Project 1. GitHub. https://github.com/dbell3/CSE-150-Proj-1 
3.4. metrics and scoring: Quantifying the quality of predictions. scikit. (n.d.). https://scikit-learn.org/stable/modules/model_evaluation.html 

### Generative AI Usage

https://chatgpt.com/?model=auto 
explanation: We mainly used ChatGpt for debugging and interpretation of our results
