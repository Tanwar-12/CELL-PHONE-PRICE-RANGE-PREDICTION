# CELL-PHONE PRICE RANGE PREDICTION
![image](https://github.com/Tanwar-12/CELL-PHONE-PRICE-RANGE-PREDICTION/assets/110081008/4a4d02fa-c934-457f-952d-aa76eda1412e)

# BUSINESS CASE:
**Find out some relation between features of a mobile phone(eg:- RAM, Internal Memory etc) and its selling price. In this problem you do not have to predict the actual price but a price range indicating how high the price is.**

## IMPORTING THE PYHTON LIBRARIES:
* NUMPY
* PANDAS
* MATPLOTLIB
* SEABORN
  
## LOAD THE DATASET

# DOMAIN ANALYSIS

  #### INPUT VARIABLES :

* battery_power = Total energy a battery can store in one time measured in mAh(Continuous)
* blue = Has bluetooth or not (Categorical)
* clock_speed = speed at which microprocessor executes instructions(Continuous)
* dual_sim = Has dual sim support or not (Categorical)
* fc = Front Camera mega pixels(Continuous)
* four_g = Has 4G or not (Categorical)
* int_memory = Internal Memory in Gigabytes(Continuous)
* m_dep = Mobile Depth in cm(Continuous)
* mobile_wt = Weight of mobile phone(Continuous)
* n_cores = Number of cores of processor(Continuous)
* pc = Primary Camera mega pixels(Continuous)
* px_height = Pixel Resolution Height(Continuous)
* px_weight = Pixel Resolution Weight(Continuous)
* ram = Random Access Memory in Megabytes(Continuous)
* sc_h = Screen Height of mobile in cm(Continuous)
* sc_w = Screen Width of mobile in cm(Continuous)
* talk_time = longest time that a single battery charge will last when you are(Continuous)
* three_g = Has 3G or not (Categorical)
* touch_screen = Has touch screen or not (Categorical)
* wifi = Has wifi or not(Categorical)

#### OUTPUT VARIABLE 

* Price Range = This is the target variable with value of 0(low cost), 1(medium cost), 2(high cost) and 3(very high cost).

  ### BASIC CHECKS
  
  * First 5 records of dataset
  * Last 5 records of dataset
  #### EXAMINE THE DATA
    
  * Data type of each column
  * Column names in dataset
  * Memory Usage
  * statistical analysis
  * CHECK THE CATEGORICAL COLUMNS
  * CHECK NORMALITY OF COLUMNS
 # EXPLORATORY DATA ANALYSIS
   ## UNIVARIATE ANALYSIS
   ![image](https://github.com/Tanwar-12/CELL-PHONE-PRICE-RANGE-PREDICTION/assets/110081008/5cc08832-35db-49eb-aca7-f8eb21e6ed34)
   ### ## INSIGHTS:
* None of the column is normally distributed.
* Distribution of fc,px_height,sc_w columns is left skewed.
* Distribution battery_power,clock_width,int_memory,m_dep,mobile_wt,pc,px_width,ram,sc_h,sc_w,talk_time columns has flat kurtosis.

  ### Check data balance
  ![image](https://github.com/Tanwar-12/CELL-PHONE-PRICE-RANGE-PREDICTION/assets/110081008/d5584c2a-1083-4fc9-a2c7-72619c800b5b)

  #### INSIGHTS:
   * data is balanced

  ## BIVARIATE ANALYSIS
  **Correlation of categorical variable with target**
  ![image](https://github.com/Tanwar-12/CELL-PHONE-PRICE-RANGE-PREDICTION/assets/110081008/a333f98b-5dbd-43d2-ba26-5916d388035c)
  ### INSIGHTS:
* Battery Power variable has correlation with target variable price range as battery power increases price of phone increases.
* internal memory variable has correlation with target variable price range as internal memory increases price of phone increases.
* px_width and px_height variable has correlation with target variable price range as px_width and px_height increases price of phone increases.
* ram has strong correlation with target variable as ram price of phone increases.
* primary camera also shows some corelation with target.
* 'clock_speed', 'fc','m_dep', 'mobile_wt', 'sc_h', 'sc_w', 'talk_time' does not have any corelation with increasing price of phone.

  ![image](https://github.com/Tanwar-12/CELL-PHONE-PRICE-RANGE-PREDICTION/assets/110081008/ed0e5c70-f609-4842-9f5b-cf6a465daaa5)
  ### INSIGHTS:
  * blue, dual_sim, n_cores, touch_screen, wifi has no correlation with target
  * three_g and four_g has correlation with target
 
  # MULTIVARIATE ANALYSIS:
  ![image](https://github.com/Tanwar-12/CELL-PHONE-PRICE-RANGE-PREDICTION/assets/110081008/1bcc6192-4c58-4fba-b10b-caddf7d7ff57)


  ## DATA PRE-PROCESSING
   * CHECK NULL VALUES
   * CHECK OUTLIERS
     ![image](https://github.com/Tanwar-12/CELL-PHONE-PRICE-RANGE-PREDICTION/assets/110081008/c6204264-d7ab-42d2-add1-79da3c55c891)
 ### HANDLING OUTLIERS
 ![image](https://github.com/Tanwar-12/CELL-PHONE-PRICE-RANGE-PREDICTION/assets/110081008/1ef771e1-c048-4cd0-89c4-3ada4f7f5b6c)
 ![image](https://github.com/Tanwar-12/CELL-PHONE-PRICE-RANGE-PREDICTION/assets/110081008/df145c78-5a24-4c76-bc51-02728ffdccb1)

 #### CHECKING CORRUPTED VALUES
 * From EDA it is observed that,in the numerical columns there are 4 columns ('fc', 'pc', 'px_height', 'sc_w') whch have few 0 entries.
* However, the variables "front camera"(fc), "primary camera"(pc) having 0 as an entry can bes assumed that the mobile doesn't have front/rear camera.
* But the other two variables "pixel height"(pc_height) and "screen_width"(sc_w) can't have 0 as their values.
* Hence, these must be marked as corrupted.
 ### SCALING
 # FEATURE ENGINEERING
 ![image](https://github.com/Tanwar-12/CELL-PHONE-PRICE-RANGE-PREDICTION/assets/110081008/da5a348a-d510-41d5-84ab-c330cfd3a575)

 **ram, pixel_height, pixel_width, battery power, these are some featues which affects on cell phone price**

**clock_speed, m_dep, touch_screen, mobile_wt has very less corelation with target**

### TRAIN -TEST SPLIT
# MODEL CREATION
## 1. LOGISTIC REGRESSION
**EVALUATION**
*Testing Accuracy 0.942*
*Training Accuracy 0.934375*

    precision    recall  f1-score   support

           0       0.99      1.00      1.00       105
           1       0.91      0.95      0.92        91
           2       0.92      0.84      0.88        92
           3       0.95      0.97      0.96       112

    accuracy                           0.94       400
    macro avg       0.94      0.94      0.94       400

 ### HYPERPARAMETER TUNING
 **Testing Accuracy 0.975**

    precision    recall  f1-score   support

           0       1.00      0.96      0.98       105
           1       0.95      1.00      0.97        91
           2       0.98      0.96      0.97        92
           3       0.97      0.98      0.98       112

    accuracy                           0.97       400
    macro avg       0.97      0.98      0.97       400
    weighted avg       0.98      0.97      0.98       400

## 2.SVM
### EVALUATION
     precision    recall  f1-score   support

           0       0.95      0.94      0.95       105
           1       0.81      0.87      0.84        91
           2       0.75      0.78      0.77        92
           3       0.93      0.85      0.89       112

    accuracy                           0.86       400
    macro avg       0.86      0.86      0.86       400
    weighted avg       0.87      0.86      0.86       400

# MODEL COMPARISON


### Model	Accuracy:

* LogisticRegression	0.9750
  
* SVM	0.9625
  
* DecisionTree	0.8350
  
* RandomForest	0.8625
   
* BaggingClassifier	0.9725
  
* XGBoost	0.8950
  
![image](https://github.com/Tanwar-12/CELL-PHONE-PRICE-RANGE-PREDICTION/assets/110081008/e504a367-dd3f-4ee7-a160-63f41634ffdf)

## FEATURE IMPORTANCE
#### USING LOGISTIC REGRESSION ALOGRITHM
![image](https://github.com/Tanwar-12/CELL-PHONE-PRICE-RANGE-PREDICTION/assets/110081008/3abefebd-01d9-4b17-aa5b-f11262c60e18)
### RESULT: FOR ALL TOP ALGORITHMS ram, battery_power, px_height, px_width are strong features.




  
