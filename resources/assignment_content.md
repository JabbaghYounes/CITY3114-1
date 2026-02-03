# CITY3114 - Machine Learning and AI
# Assignment 1

**Author:** Jabbagh Younes
**Date:** 02/04/2025

---

## Table of Contents

1. Introduction
2. Task 1: Four Fundamental Machine Learning Algorithms
   - 2.1 Decision Trees
   - 2.2 Support Vector Machines (SVM)
   - 2.3 K-Nearest Neighbours (k-NN)
   - 2.4 Naive Bayes Classifier
3. Task 2: Performance Evaluation of Decision Trees
   - 3.1 Dataset Considerations
   - 3.2 Evaluation Metrics
   - 3.3 Performance Predictions
4. Task 3: Advanced Learning Methods for Improving Fundamental Algorithms
   - 4.1 Ensemble Methods
   - 4.2 Hyperparameter Optimisation
   - 4.3 Feature Engineering and Selection
   - 4.4 Cross-Validation Techniques
5. Conclusion
6. References

---

## 1. Introduction

Machine learning has emerged as one of the most transformative technologies in contemporary computing, fundamentally altering how systems process information and make decisions. As a subset of artificial intelligence, machine learning enables computer systems to learn from data and improve their performance over time without being explicitly programmed for every possible scenario [1]. This capability has profound implications across numerous industries, from healthcare diagnostics to financial fraud detection, making an understanding of fundamental machine learning algorithms essential for modern computing professionals.

The significance of machine learning stems from its ability to identify patterns within vast datasets that would be impossible for humans to detect manually. According to Jordan and Mitchell [2], machine learning algorithms can process millions of data points to extract meaningful insights, enabling organisations to make data-driven decisions with unprecedented accuracy. This report examines four fundamental machine learning algorithms: Decision Trees, Support Vector Machines (SVM), K-Nearest Neighbours (k-NN), and Naive Bayes classifiers. Each algorithm represents a distinct approach to pattern recognition and classification, with unique strengths and limitations that make them suitable for different applications.

The objectives of this report are threefold. First, it provides an in-depth analysis of each algorithm's mechanics, real-world applications, feature requirements, and associated challenges. Second, it presents a comprehensive framework for evaluating the performance of Decision Trees on real-world problems, including dataset considerations and evaluation metrics. Third, it explores advanced learning methods that can enhance the performance of these fundamental algorithms, demonstrating how modern techniques can address their inherent limitations.

---

## 2. Task 1: Four Fundamental Machine Learning Algorithms

### 2.1 Decision Trees

#### Mechanics of the Algorithm

Decision Trees are supervised learning algorithms that create a tree-like model of decisions based on feature values. The algorithm works by recursively partitioning the feature space into regions, with each internal node representing a test on a feature, each branch representing the outcome of that test, and each leaf node representing a class label or prediction [3].

The construction of a Decision Tree involves selecting the best feature to split the data at each node. This selection is typically based on metrics such as Information Gain (using entropy) or Gini Impurity. Information Gain measures the reduction in entropy achieved by splitting on a particular feature, calculated as:

**Information Gain = Entropy(parent) - Weighted Average of Entropy(children)**

Where entropy is defined as: **H(S) = -Σ p(i) log₂ p(i)** for each class i in the dataset [4].

The algorithm continues splitting until a stopping criterion is met, such as maximum depth, minimum samples per leaf, or when all samples in a node belong to the same class.

#### Real-World Application

Decision Trees are extensively used in medical diagnosis systems. A prominent application is in the diagnosis of cardiovascular diseases, where the algorithm analyses patient features to predict the likelihood of heart disease [5].

#### Features Used in This Application

In cardiovascular disease diagnosis, the features typically include:
- **Demographic features:** Age, gender
- **Clinical measurements:** Blood pressure (systolic and diastolic), cholesterol levels, resting heart rate
- **Lifestyle factors:** Smoking status, physical activity levels
- **Medical history:** Presence of diabetes, family history of heart disease
- **Diagnostic test results:** ECG readings, chest pain type, exercise-induced angina [6]

#### Challenges

Decision Trees face several challenges. **Overfitting** is a primary concern, where the tree becomes too complex and captures noise in the training data rather than underlying patterns [7]. Additionally, Decision Trees are **sensitive to small variations** in the data, where minor changes can result in completely different tree structures. They also exhibit **bias towards features with more levels**, potentially overlooking important binary features. Finally, Decision Trees struggle with **capturing linear relationships** between features, as they create axis-parallel decision boundaries [8].

### 2.2 Support Vector Machines (SVM)

#### Mechanics of the Algorithm

Support Vector Machines are powerful supervised learning algorithms that find the optimal hyperplane to separate different classes in the feature space. The fundamental principle is to maximise the margin between the closest points of different classes, known as support vectors [9].

For linearly separable data, SVM finds the hyperplane **w·x + b = 0** that maximises the margin 2/||w||. The optimisation problem is formulated as:

**Minimise: ½||w||²**
**Subject to: yᵢ(w·xᵢ + b) ≥ 1 for all i**

For non-linearly separable data, SVM employs the **kernel trick**, which maps the original features into a higher-dimensional space where linear separation becomes possible. Common kernels include the Radial Basis Function (RBF), polynomial kernel, and sigmoid kernel [10].

#### Real-World Application

SVMs are widely used in **email spam detection systems**. Major email providers utilise SVM-based classifiers to automatically filter spam messages from legitimate emails [11].

#### Features Used in This Application

For spam detection, the features include:
- **Text-based features:** Word frequencies, presence of specific keywords (e.g., "free", "winner", "urgent")
- **Structural features:** Email length, number of links, presence of attachments
- **Header information:** Sender reputation, domain age, authentication results (SPF, DKIM)
- **Metadata:** Time of sending, geographical origin
- **N-gram features:** Sequences of words that commonly appear in spam [12]

#### Challenges

SVM challenges include **computational complexity** with large datasets, as the training time scales between O(n²) and O(n³) depending on the implementation [13]. **Kernel selection** is non-trivial and significantly impacts performance. SVMs are also **sensitive to feature scaling**, requiring normalisation of input features. Additionally, they **lack probabilistic outputs** by default, making confidence estimation difficult, and the resulting models can be **difficult to interpret** compared to Decision Trees [14].

### 2.3 K-Nearest Neighbours (k-NN)

#### Mechanics of the Algorithm

K-Nearest Neighbours is an instance-based learning algorithm that classifies new instances based on the majority class among its k closest training examples. Unlike parametric methods, k-NN makes no assumptions about the underlying data distribution, making it a non-parametric approach [15].

The algorithm operates by:
1. Calculating the distance between the query instance and all training instances
2. Selecting the k instances with the smallest distances
3. Assigning the class based on majority voting (for classification) or averaging (for regression)

Distance metrics commonly used include **Euclidean distance**: d(x,y) = √Σ(xᵢ - yᵢ)², **Manhattan distance**: d(x,y) = Σ|xᵢ - yᵢ|, and **Minkowski distance** as a generalisation [16].

#### Real-World Application

K-NN is prominently used in **recommendation systems**, particularly for collaborative filtering in platforms like Netflix and Amazon. The algorithm identifies users with similar viewing or purchasing patterns to recommend new content [17].

#### Features Used in This Application

In recommendation systems, features include:
- **User behaviour data:** Viewing history, ratings given, time spent on items
- **Item attributes:** Genre, director, cast, release year (for movies)
- **Interaction patterns:** Click-through rates, purchase history, wishlist additions
- **Temporal features:** Time of day, seasonal preferences
- **Demographic information:** Age group, location, subscription tier [18]

#### Challenges

K-NN faces the **curse of dimensionality**, where performance degrades significantly as the number of features increases because distances become less meaningful in high-dimensional spaces [19]. The algorithm has **high computational cost at prediction time**, as it must calculate distances to all training points. **Storage requirements** are substantial since the entire training dataset must be retained. **Choosing the optimal k** value is challenging, and k-NN is **sensitive to irrelevant features** and different scales, requiring careful preprocessing [20].

### 2.4 Naive Bayes Classifier

#### Mechanics of the Algorithm

Naive Bayes is a probabilistic classifier based on Bayes' theorem with the "naive" assumption of conditional independence between features given the class label. Despite this strong assumption, it performs remarkably well in many real-world applications [21].

Bayes' theorem states: **P(C|X) = P(X|C) × P(C) / P(X)**

Where P(C|X) is the posterior probability of class C given features X, P(X|C) is the likelihood, P(C) is the prior probability, and P(X) is the evidence.

The naive independence assumption simplifies this to:
**P(C|x₁,...,xₙ) ∝ P(C) × Π P(xᵢ|C)**

Variants include **Gaussian Naive Bayes** (for continuous features assuming normal distribution), **Multinomial Naive Bayes** (for discrete count features), and **Bernoulli Naive Bayes** (for binary features) [22].

#### Real-World Application

Naive Bayes is extensively used in **sentiment analysis** for social media monitoring. Companies use it to automatically classify customer opinions expressed on platforms like Twitter and Facebook as positive, negative, or neutral [23].

#### Features Used in This Application

For sentiment analysis, the features comprise:
- **Bag-of-words representation:** Frequency counts of individual words
- **TF-IDF scores:** Term frequency-inverse document frequency weightings
- **N-grams:** Bigrams and trigrams capturing phrase-level sentiment
- **Part-of-speech tags:** Adjectives and adverbs often carry sentiment
- **Negation handling:** Detection of negation words that flip sentiment
- **Emoticons and emojis:** Direct sentiment indicators [24]

#### Challenges

The **independence assumption is often violated** in practice, as words in natural language exhibit strong dependencies. **Zero-frequency problem** occurs when a feature value not seen during training appears at test time, requiring smoothing techniques like Laplace smoothing [25]. Naive Bayes is **sensitive to feature selection**, as irrelevant features can significantly degrade performance. The algorithm **cannot capture feature interactions** due to its independence assumption, and it produces **poorly calibrated probability estimates**, though the rankings remain useful [26].

---

## 3. Task 2: Performance Evaluation of Decision Trees

### 3.1 Dataset Considerations

To evaluate Decision Tree performance on cardiovascular disease diagnosis, careful consideration of the dataset is essential. The **UCI Heart Disease dataset** is a widely-used benchmark containing 303 instances from the Cleveland Clinic, with 14 attributes including age, sex, chest pain type, resting blood pressure, serum cholesterol, fasting blood sugar, resting ECG results, maximum heart rate achieved, exercise-induced angina, ST depression, slope of peak exercise ST segment, number of major vessels, thalassemia, and the target variable indicating presence of heart disease [27].

**Dataset Size and Format:** The dataset is relatively small by modern standards but sufficient for Decision Tree evaluation. It is stored in CSV format with mixed data types (numerical and categorical). The feature space is 13-dimensional after excluding the target variable.

**Data Organisation:** The data requires preprocessing including:
- **Missing value handling:** Approximately 6 instances contain missing values, typically handled through imputation or removal
- **Feature encoding:** Categorical variables require one-hot encoding or label encoding
- **Train-test split:** Typically 70-80% training, 20-30% testing, with stratification to maintain class distribution
- **Feature scaling:** While not strictly necessary for Decision Trees, it facilitates comparison with other algorithms [28]

For more robust evaluation, larger datasets such as the **Kaggle Cardiovascular Disease dataset** (70,000 instances) or the **Framingham Heart Study dataset** provide more comprehensive evaluation opportunities [29].

### 3.2 Evaluation Metrics

#### Accuracy

**Definition:** The proportion of correct predictions among total predictions.

**Formula:** Accuracy = (TP + TN) / (TP + TN + FP + FN)

Where TP = True Positives, TN = True Negatives, FP = False Positives, FN = False Negatives.

**Measurement:** Calculate after predictions on the test set by comparing predicted labels to actual labels.

**Interpretation:** While intuitive, accuracy can be misleading for imbalanced datasets. In heart disease diagnosis, where the cost of false negatives is high, accuracy alone is insufficient [30].

#### Precision

**Definition:** The proportion of positive predictions that are actually correct.

**Formula:** Precision = TP / (TP + FP)

**Measurement:** Count true positive predictions and divide by all positive predictions.

**Interpretation:** High precision indicates low false positive rate. In medical diagnosis, high precision means few healthy patients are incorrectly diagnosed with heart disease [31].

#### Recall (Sensitivity)

**Definition:** The proportion of actual positive cases correctly identified.

**Formula:** Recall = TP / (TP + FN)

**Measurement:** Count true positives and divide by all actual positive cases.

**Interpretation:** Crucial in medical applications where missing a positive case (disease) has severe consequences. High recall ensures most patients with heart disease are detected [31].

#### F1-Score

**Definition:** The harmonic mean of precision and recall, providing a balanced measure.

**Formula:** F1 = 2 × (Precision × Recall) / (Precision + Recall)

**Measurement:** Calculate precision and recall first, then apply the formula.

**Interpretation:** Useful when seeking balance between precision and recall, particularly for imbalanced datasets [32].

#### Area Under ROC Curve (AUC-ROC)

**Definition:** Measures the classifier's ability to distinguish between classes across all threshold values.

**Formula:** AUC = ∫ TPR d(FPR), where TPR is True Positive Rate and FPR is False Positive Rate.

**Measurement:** Plot ROC curve by varying the classification threshold and calculate the area underneath.

**Interpretation:** AUC of 0.5 indicates random guessing, while 1.0 indicates perfect classification. Values above 0.8 are generally considered good for medical applications [33].

#### Confusion Matrix

The confusion matrix provides a comprehensive view of classifier performance, showing the distribution of predictions across actual classes. It reveals patterns of misclassification that scalar metrics cannot capture [34].

### 3.3 Performance Predictions

Based on published research, Decision Trees demonstrate moderate to good performance on cardiovascular disease classification tasks. Mohan, Thirumalai, and Srivastava [35] reported an accuracy of 88.7% using Decision Tree classification on the Cleveland Heart Disease dataset with optimised hyperparameters.

A comprehensive study by Rani [36] comparing multiple algorithms found Decision Trees achieved:
- **Accuracy:** 81.97%
- **Precision:** 82.35%
- **Recall:** 84.85%
- **F1-Score:** 83.58%

Research published in IEEE by Patel et al. [37] evaluated Decision Trees (C4.5 algorithm) on heart disease prediction, achieving:
- **Accuracy:** 84.1%
- **AUC-ROC:** 0.85

**Predicted Real-World Performance:** Based on the literature review, Decision Trees are expected to achieve approximately **82-89% accuracy** on cardiovascular disease diagnosis, with recall values around **84-87%**, which is critical for ensuring most disease cases are detected. However, performance varies based on:

1. **Dataset quality:** Larger, more diverse datasets generally yield better generalisation
2. **Feature engineering:** Careful feature selection can improve performance by 5-10% [38]
3. **Hyperparameter tuning:** Optimising tree depth, minimum samples per leaf, and pruning strategies significantly impacts results
4. **Class imbalance handling:** Techniques like SMOTE or class weighting can improve minority class detection [39]

The predicted performance makes Decision Trees viable for clinical decision support, though they should complement rather than replace expert medical judgement due to the inherent limitations and the critical nature of medical diagnoses.

---

## 4. Task 3: Advanced Learning Methods for Improving Fundamental Algorithms

### 4.1 Ensemble Methods

Ensemble methods combine multiple models to achieve better performance than any single model alone. These techniques can substantially improve the fundamental algorithms discussed above.

**Random Forests** extend Decision Trees by constructing multiple trees on random subsets of features and data, then aggregating predictions through voting. Breiman [40] demonstrated that Random Forests reduce variance without increasing bias, addressing the overfitting tendency of individual Decision Trees. Studies show Random Forests can improve accuracy by 10-15% over single Decision Trees on classification tasks [41].

**Gradient Boosting** sequentially builds models, with each new model correcting errors of the previous ensemble. XGBoost, a popular implementation, has achieved state-of-the-art results across numerous Kaggle competitions. Chen and Guestrin [42] showed that gradient boosting can improve base learner performance by 5-20% depending on the dataset.

**Bagging (Bootstrap Aggregating)** reduces variance by training multiple models on bootstrapped samples. This technique is particularly effective for high-variance algorithms like k-NN and Decision Trees, improving stability and generalisation [43].

### 4.2 Hyperparameter Optimisation

Systematic hyperparameter tuning can significantly enhance algorithm performance.

**Grid Search** exhaustively evaluates all combinations of specified hyperparameter values. While computationally expensive, it guarantees finding the optimal combination within the search space. For SVMs, tuning the regularisation parameter C and kernel parameters can improve accuracy by 5-15% [44].

**Random Search** samples hyperparameter combinations randomly, often achieving comparable results to grid search with significantly less computation. Bergstra and Bengio [45] demonstrated that random search is more efficient for high-dimensional hyperparameter spaces.

**Bayesian Optimisation** uses probabilistic models to intelligently select hyperparameters to evaluate, focusing on promising regions of the search space. This approach is particularly valuable for computationally expensive models, reducing the number of evaluations needed to find optimal configurations [46].

### 4.3 Feature Engineering and Selection

Feature quality directly impacts algorithm performance across all fundamental methods.

**Feature Selection** methods include filter methods (correlation-based), wrapper methods (recursive feature elimination), and embedded methods (L1 regularisation). Guyon and Elisseeff [47] demonstrated that removing irrelevant features can improve k-NN performance by 20-30% while reducing computational cost.

**Dimensionality Reduction** techniques like Principal Component Analysis (PCA) and t-SNE can address the curse of dimensionality affecting k-NN. By projecting data to lower dimensions while preserving variance, these techniques improve both performance and efficiency [48].

**Feature Transformation** including polynomial features, logarithmic transformations, and interaction terms can help algorithms capture non-linear relationships. For Naive Bayes, discretisation of continuous features can improve performance when the Gaussian assumption is violated [49].

### 4.4 Cross-Validation Techniques

Robust evaluation methodologies ensure reliable performance estimates and prevent overfitting during model selection.

**K-Fold Cross-Validation** divides data into k subsets, training on k-1 folds and validating on the remaining fold, rotating through all combinations. This provides more reliable performance estimates than single train-test splits, particularly for small datasets [50].

**Stratified Cross-Validation** maintains class proportions in each fold, essential for imbalanced datasets common in medical applications.

**Nested Cross-Validation** uses an outer loop for performance estimation and an inner loop for hyperparameter tuning, providing unbiased performance estimates even when hyperparameters are optimised [51].

These advanced methods, when applied appropriately, can transform the fundamental algorithms from baseline classifiers into competitive systems suitable for production deployment. The choice of improvement strategy depends on the specific limitations of the base algorithm and the characteristics of the problem domain.

---

## 5. Conclusion

This report has provided a comprehensive analysis of four fundamental machine learning algorithms: Decision Trees, Support Vector Machines, K-Nearest Neighbours, and Naive Bayes classifiers. Each algorithm offers unique advantages and faces distinct challenges that influence their suitability for different applications.

Decision Trees offer interpretability and ease of use but are prone to overfitting and instability. SVMs provide robust performance in high-dimensional spaces but face computational challenges with large datasets. K-NN's simplicity and non-parametric nature make it versatile, though the curse of dimensionality and prediction-time costs limit scalability. Naive Bayes achieves remarkable efficiency despite its simplifying assumptions, making it particularly effective for text classification tasks.

The evaluation framework presented for Decision Trees demonstrates the importance of selecting appropriate metrics beyond simple accuracy. For medical applications like cardiovascular disease diagnosis, recall and the balance captured by F1-score are crucial considerations. Published research suggests Decision Trees can achieve approximately 82-89% accuracy on heart disease datasets, with careful attention to data preprocessing, feature engineering, and hyperparameter tuning.

Advanced learning methods offer substantial opportunities for improving fundamental algorithm performance. Ensemble methods like Random Forests and Gradient Boosting address individual model weaknesses through aggregation. Systematic hyperparameter optimisation can yield significant improvements with relatively little additional complexity. Feature engineering and selection directly address the quality of input data, while robust cross-validation ensures reliable performance estimates.

Future developments in machine learning will likely see continued integration of these fundamental algorithms with deep learning approaches, potentially combining the interpretability of classical methods with the representational power of neural networks. Understanding these foundational techniques remains essential for practitioners seeking to apply machine learning effectively to real-world problems.

---

## 6. References

[1] T. M. Mitchell, *Machine Learning*. New York, NY, USA: McGraw-Hill, 1997.

[2] M. I. Jordan and T. M. Mitchell, "Machine learning: Trends, perspectives, and prospects," *Science*, vol. 349, no. 6245, pp. 255-260, 2015.

[3] J. R. Quinlan, *C4.5: Programs for Machine Learning*. San Francisco, CA, USA: Morgan Kaufmann, 1993.

[4] L. Breiman, J. Friedman, C. J. Stone, and R. A. Olshen, *Classification and Regression Trees*. Boca Raton, FL, USA: CRC Press, 1984.

[5] R. Detrano et al., "International application of a new probability algorithm for the diagnosis of coronary artery disease," *American Journal of Cardiology*, vol. 64, no. 5, pp. 304-310, 1989.

[6] A. Javaid et al., "Medicine 2032: The future of cardiovascular disease prevention with machine learning and digital health technology," *American Journal of Preventive Cardiology*, vol. 12, p. 100379, 2022.

[7] T. Hastie, R. Tibshirani, and J. Friedman, *The Elements of Statistical Learning*, 2nd ed. New York, NY, USA: Springer, 2009.

[8] P. Domingos, "A few useful things to know about machine learning," *Communications of the ACM*, vol. 55, no. 10, pp. 78-87, 2012.

[9] C. Cortes and V. Vapnik, "Support-vector networks," *Machine Learning*, vol. 20, no. 3, pp. 273-297, 1995.

[10] B. Scholkopf and A. J. Smola, *Learning with Kernels: Support Vector Machines, Regularization, Optimization, and Beyond*. Cambridge, MA, USA: MIT Press, 2002.

[11] A. Kolcz and J. Alspector, "SVM-based filtering of e-mail spam with content-specific misclassification costs," in *Proc. IEEE Int. Conf. Data Mining Workshop on Text Mining*, 2001, pp. 1-8.

[12] G. V. Cormack, "Email spam filtering: A systematic review," *Foundations and Trends in Information Retrieval*, vol. 1, no. 4, pp. 335-455, 2008.

[13] J. Platt, "Sequential minimal optimization: A fast algorithm for training support vector machines," Microsoft Research, Tech. Rep. MSR-TR-98-14, 1998.

[14] C.-W. Hsu, C.-C. Chang, and C.-J. Lin, "A practical guide to support vector classification," National Taiwan University, Tech. Rep., 2003.

[15] T. Cover and P. Hart, "Nearest neighbor pattern classification," *IEEE Transactions on Information Theory*, vol. 13, no. 1, pp. 21-27, 1967.

[16] D. Aha, D. Kibler, and M. Albert, "Instance-based learning algorithms," *Machine Learning*, vol. 6, no. 1, pp. 37-66, 1991.

[17] X. Su and T. M. Khoshgoftaar, "A survey of collaborative filtering techniques," *Advances in Artificial Intelligence*, vol. 2009, pp. 1-19, 2009.

[18] F. Ricci, L. Rokach, and B. Shapira, *Recommender Systems Handbook*, 2nd ed. New York, NY, USA: Springer, 2015.

[19] R. Bellman, *Dynamic Programming*. Princeton, NJ, USA: Princeton University Press, 1957.

[20] K. Beyer, J. Goldstein, R. Ramakrishnan, and U. Shaft, "When is 'nearest neighbor' meaningful?," in *Proc. 7th Int. Conf. Database Theory*, 1999, pp. 217-235.

[21] D. D. Lewis, "Naive (Bayes) at forty: The independence assumption in information retrieval," in *Proc. 10th European Conf. Machine Learning*, 1998, pp. 4-15.

[22] C. D. Manning, P. Raghavan, and H. Schutze, *Introduction to Information Retrieval*. Cambridge, UK: Cambridge University Press, 2008.

[23] B. Pang and L. Lee, "Opinion mining and sentiment analysis," *Foundations and Trends in Information Retrieval*, vol. 2, no. 1-2, pp. 1-135, 2008.

[24] B. Liu, *Sentiment Analysis and Opinion Mining*. San Rafael, CA, USA: Morgan & Claypool, 2012.

[25] A. McCallum and K. Nigam, "A comparison of event models for naive Bayes text classification," in *Proc. AAAI-98 Workshop on Learning for Text Categorization*, 1998, pp. 41-48.

[26] H. Zhang, "The optimality of naive Bayes," in *Proc. 17th Int. Florida Artificial Intelligence Research Society Conf.*, 2004, pp. 562-567.

[27] D. Dua and C. Graff, "UCI Machine Learning Repository," University of California, Irvine, 2019. [Online]. Available: http://archive.ics.uci.edu/ml

[28] S. Raschka and V. Mirjalili, *Python Machine Learning*, 3rd ed. Birmingham, UK: Packt Publishing, 2019.

[29] P. W. F. Wilson et al., "Prediction of coronary heart disease using risk factor categories," *Circulation*, vol. 97, no. 18, pp. 1837-1847, 1998.

[30] M. Sokolova and G. Lapalme, "A systematic analysis of performance measures for classification tasks," *Information Processing & Management*, vol. 45, no. 4, pp. 427-437, 2009.

[31] D. M. W. Powers, "Evaluation: From precision, recall and F-measure to ROC, informedness, markedness and correlation," *Journal of Machine Learning Technologies*, vol. 2, no. 1, pp. 37-63, 2011.

[32] C. Goutte and E. Gaussier, "A probabilistic interpretation of precision, recall and F-score, with implication for evaluation," in *Proc. 27th European Conf. Information Retrieval*, 2005, pp. 345-359.

[33] A. P. Bradley, "The use of the area under the ROC curve in the evaluation of machine learning algorithms," *Pattern Recognition*, vol. 30, no. 7, pp. 1145-1159, 1997.

[34] S. Visa, B. Ramsay, A. L. Ralescu, and E. van der Knaap, "Confusion matrix-based feature selection," in *Proc. 22nd Midwest Artificial Intelligence and Cognitive Science Conf.*, 2011, pp. 120-127.

[35] S. Mohan, C. Thirumalai, and G. Srivastava, "Effective heart disease prediction using hybrid machine learning techniques," *IEEE Access*, vol. 7, pp. 81542-81554, 2019.

[36] K. U. Rani, "Analysis of heart diseases dataset using neural network approach," *International Journal of Data Mining & Knowledge Management Process*, vol. 1, no. 5, pp. 1-8, 2011.

[37] J. Patel, D. TejalUpadhyay, and S. Patel, "Heart disease prediction using machine learning and data mining technique," *International Journal of Computer Science and Communication*, vol. 7, no. 1, pp. 129-137, 2016.

[38] G. Chandrashekar and F. Sahin, "A survey on feature selection methods," *Computers & Electrical Engineering*, vol. 40, no. 1, pp. 16-28, 2014.

[39] N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer, "SMOTE: Synthetic minority over-sampling technique," *Journal of Artificial Intelligence Research*, vol. 16, pp. 321-357, 2002.

[40] L. Breiman, "Random forests," *Machine Learning*, vol. 45, no. 1, pp. 5-32, 2001.

[41] M. Fernandez-Delgado, E. Cernadas, S. Barro, and D. Amorim, "Do we need hundreds of classifiers to solve real world classification problems?," *Journal of Machine Learning Research*, vol. 15, no. 1, pp. 3133-3181, 2014.

[42] T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in *Proc. 22nd ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining*, 2016, pp. 785-794.

[43] L. Breiman, "Bagging predictors," *Machine Learning*, vol. 24, no. 2, pp. 123-140, 1996.

[44] C.-C. Chang and C.-J. Lin, "LIBSVM: A library for support vector machines," *ACM Transactions on Intelligent Systems and Technology*, vol. 2, no. 3, pp. 1-27, 2011.

[45] J. Bergstra and Y. Bengio, "Random search for hyper-parameter optimization," *Journal of Machine Learning Research*, vol. 13, pp. 281-305, 2012.

[46] J. Snoek, H. Larochelle, and R. P. Adams, "Practical Bayesian optimization of machine learning algorithms," in *Proc. Advances in Neural Information Processing Systems*, 2012, pp. 2951-2959.

[47] I. Guyon and A. Elisseeff, "An introduction to variable and feature selection," *Journal of Machine Learning Research*, vol. 3, pp. 1157-1182, 2003.

[48] L. van der Maaten and G. Hinton, "Visualizing data using t-SNE," *Journal of Machine Learning Research*, vol. 9, pp. 2579-2605, 2008.

[49] U. M. Fayyad and K. B. Irani, "Multi-interval discretization of continuous-valued attributes for classification learning," in *Proc. 13th Int. Joint Conf. Artificial Intelligence*, 1993, pp. 1022-1027.

[50] R. Kohavi, "A study of cross-validation and bootstrap for accuracy estimation and model selection," in *Proc. 14th Int. Joint Conf. Artificial Intelligence*, 1995, pp. 1137-1143.

[51] G. C. Cawley and N. L. C. Talbot, "On over-fitting in model selection and subsequent selection bias in performance evaluation," *Journal of Machine Learning Research*, vol. 11, pp. 2079-2107, 2010.
