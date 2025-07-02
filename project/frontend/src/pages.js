// pages.jsx
import React from "react";
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { darcula } from 'react-syntax-highlighter/dist/esm/styles/prism';


const codeStyle = {
  background: "#1e1e1e",
  color: "#f8f8f2",
  fontFamily: "monospace",
  borderRadius: "6px",
  overflowX: "auto",
  whiteSpace: "pre",
  border: "1px solid #333",
  fontSize: "14px",
};


// ✅ Export this array
export const pages = [
  {
    title: "Introduction: Building ML Models",
    content: (
      <>
        <p>Welcome to the Building ML Models lab! In this lab, you’ll take a hands on journey through the lifecycle of developing, evaluating, and improving machine learning models for a binary classification task. 
          You’ll begin by loading and exploring a synthetic dataset that closely mirrors real-world structure. From there, you’ll train baseline and alternative models, analyze their performance, diagnose errors, 
          and recommend data driven improvements, just like in an end to end ML project.Whether you're a software developer exploring ML in production or a product manager looking to understand how models make decisions,
          this lab is designed to give you practical,transferable experience in constructing, comparing, and refining machine learning models.
        </p>
       <hr style={{ borderTop: "5px solid rgb(241, 239, 239)", margin: "20px 0" }} />
        <h3>What You’ll Learn</h3>
        <p> By the end of this lab, you’ll be able to: <br></br>
        
      Build and evaluate a baseline ML model to establish a performance reference point.
      Compare multiple models (e.g., logistic regression, SVM, random forest) to assess relative strengths and weaknesses.
      Track and document model configurations and results using structured metrics and summaries.
      Analyze model errors (e.g., confusion matrices, misclassification patterns) to guide improvements and define next steps.
        </p>
        <hr style={{ borderTop: "5px solid rgb(241, 239, 239)", margin: "20px 0" }} />
        <h3>Why It Matters</h3>
        <p>In machine learning, building a model is just one part of the process. Great models:</p><br></br>
         <ul>
          <li>Generalize well to new, unseen data</li>
          <li>Handle class imbalances without bias</li>
          <li>Balance performance with interpretability</li>  
          <li> Align with practical business or product goals </li>
        </ul>
        <p>This lab reflects real world modeling workflows, where the journey involves iteration, critical thinking, and communication, not just chasing accuracy.</p>
        <hr style={{ borderTop: "5px solid rgb(241, 239, 239)", margin: "20px 0" }} />
        <h3>Dataset</h3>
        <p>To keep the focus on modeling, this lab uses a synthetic binary classification dataset created with make_classification() from scikit-learn. This dataset:</p><br></br>
         <ul>
          <li>Includes multiple informative and redundant numeric features</li>
          <li>Simulates rea -world data structure (noise, correlation, and imbalance)</li>
          <li>Contains a binary label: 0 or 1</li>  
        </ul>
        <p>This setup allows us to explore realistic modeling challenges without distractions from messy preprocessing.</p>
        <hr style={{ borderTop: "5px solid rgb(241, 239, 239)", margin: "20px 0" }} />
        <p>You'll begin by loading and exploring the dataset, checking for class imbalance, scaling the features, and preparing the data for modeling in PyTorch.</p>
        <hr style={{ borderTop: "5px solid rgb(241, 239, 239)", margin: "20px 0" }} />
    </>
  ),
},

{
  title: "",

    content: (
      <>
      <h3>Step 1: Load and Explore the Dataset</h3>
<p>
  Welcome to Step 1! Before you can build or evaluate a machine learning model, you need to understand the data.
  In this step, you’ll load a binary classification dataset, examine its structure, check for imbalances, and prepare
  it for modeling by normalizing features and splitting into training and testing sets.
</p>
<hr style={{ borderTop: "5px solid rgb(241, 239, 239)", margin: "20px 0" }} />
<h3>Why This Step Is Important</h3>
<p>
  The quality of your machine learning results starts with the quality of your dataset understanding.
  Jumping into modeling without first checking class balance, feature ranges, or input scales can lead to skewed results
  and models that don't generalize well.
</p>
<hr style={{ borderTop: "5px solid rgb(241, 239, 239)", margin: "20px 0" }} />
<h3>Here’s what you'll aim to do:</h3>
<ul>
  <li>Explore class distribution – Know if one class dominates (which might bias the model).</li>
  <li>Understand feature ranges – Helps guide normalization and detect outliers.</li>
  <li>Prepare train/test sets – Ensures fair model evaluation on unseen data.</li>
  <li>Normalize input features – Prevents any one feature from disproportionately influencing results.</li>
</ul>
<hr style={{ borderTop: "5px solid rgb(241, 239, 239)", margin: "20px 0" }} />
<h3>Dataset Overview</h3>
<p>
  In this lab, you'll use a binary classification dataset (<code>make_classification</code> for demo purposes,
  but easily swapped for real data). The dataset has:
</p>
<ul>
  <li>Numeric input features</li>
  <li>A binary label: 0 or 1</li>
</ul>
<hr style={{ borderTop: "5px solid rgb(241, 239, 239)", margin: "20px 0" }} />
<h3>Step 1.1 – Load and Preview the Data</h3>
<div className="code-scroll-container mt-2 p-3" style={codeStyle}>
  <code>
    <SyntaxHighlighter language="python" style={darcula}>
{`from sklearn.datasets import make_classification
import pandas as pd
import numpy as np

# Create synthetic binary classification dataset
X, y = make_classification(n_samples=1000, n_features=10, 
                           n_informative=5, n_redundant=2, 
                           n_classes=2, random_state=42)

# Convert to DataFrame for exploration
df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
df['label'] = y

# Preview the data
print(df.head())`}
</SyntaxHighlighter>
  </code>
  
</div>
<hr style={{ borderTop: "5px solid rgb(241, 239, 239)", margin: "20px 0" }} />
<h3>Step 1.2 – Explore Class Balance and Feature Ranges</h3>
<div className="code-scroll-container mt-2 p-3" style={codeStyle}>
  <code>
    <SyntaxHighlighter language="python" style={darcula}>

{`# Check class distribution
print("Class distribution:")
print(df['label'].value_counts())

# Summary stats for features
print("\\nFeature summary:")
print(df.describe())`}
</SyntaxHighlighter>
  </code>
</div>
<p>Look out for:</p>
<ul>
  <li>Whether one class is significantly more common (imbalance)</li>
  <li>Whether features vary widely in scale (e.g., one is 0–1, another 0–1000)</li>
</ul>
<hr style={{ borderTop: "5px solid rgb(241, 239, 239)", margin: "20px 0" }} />
<h3>Step 1.3 – Normalize and Split the Data</h3>
<div className="code-scroll-container mt-2 p-3" style={codeStyle}>
  <code>
    <SyntaxHighlighter language="python" style={darcula}>
{`from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch

# Separate features and labels
X = df.drop('label', axis=1).values
y = df['label'].values

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

print("Training set shape:", X_train_tensor.shape)
print("Testing set shape:", X_test_tensor.shape)`}
</SyntaxHighlighter>
  </code>
</div>
<hr style={{ borderTop: "5px solid rgb(241, 239, 239)", margin: "20px 0" }} />
<h3>Notes</h3>
<ul>
  <li><code>make_classification()</code> simulates a realistic binary classification dataset.</li>
  <li><code>StandardScaler()</code> centers and scales features to standard normal distribution (mean = 0, std = 1).</li>
  <li><code>train_test_split()</code> splits the data into separate sets, with <code>stratify=y</code> to maintain label proportions.</li>
  <li><code>torch.tensor()</code> prepares the data for PyTorch models.</li>
</ul>
<p>This ensures your data is fair, balanced, and ready for training across multiple model types.</p>
<hr style={{ borderTop: "5px solid rgb(241, 239, 239)", margin: "20px 0" }} />
<h3>Analysis</h3>
<p>Think of this like packing your gear for a trip:</p>
<ul>
  <li>First, you check that everything is balanced and fits in the bag (label distribution and scale).</li>
  <li>You organize and label your items (features and labels).</li>
  <li>Finally, you separate what goes in your carry-on vs checked luggage — that’s your train/test split!</li>
</ul>
<p>Skipping these steps could leave you unprepared when the real journey (training) begins.</p>
<hr style={{ borderTop: "5px solid rgb(241, 239, 239)", margin: "20px 0" }} />
<h3>Comprehension Check</h3>
<ul>
  <li>Why is it important to explore class balance before training?</li>
  <li>What does <code>StandardScaler</code> do, and why is it helpful?</li>
  <li>Why convert inputs to PyTorch tensors?</li>
</ul>

<hr style={{ borderTop: "5px solid rgb(241, 239, 239)", margin: "20px 0" }} />

  </>
  ),
},
{
  title: "",
  content: (
    <>
    <h3>Step 2: Build and Evaluate a Baseline Model</h3>
<p>
  Now that your dataset is loaded, cleaned, and ready to go, it's time to train a baseline model. This step is all about
  setting a reference point. You’ll train a simple logistic regression model, evaluate its performance, and record key metrics
  like accuracy, precision, and recall.
</p>
<hr style={{ borderTop: "5px solid rgb(241, 239, 239)", margin: "20px 0" }} />
<h3>Why This Step Is Important</h3>
<p>Every ML workflow should start with a baseline. Why?</p>
<ul>
  <li>It gives you a reference performance to beat with more complex models.</li>
  <li>It helps you quickly spot data issues—if a baseline performs poorly, it could indicate deeper problems.</li>
  <li>It’s often surprisingly effective, especially when interpretability and speed matter.</li>
</ul>
<p>A baseline is not about perfection, it’s about perspective.</p>
<hr style={{ borderTop: "5px solid rgb(241, 239, 239)", margin: "20px 0" }} />
<h3>Step 2.1 – Train a Logistic Regression Model</h3>
<p>
  You'll use <code>sklearn.linear_model.LogisticRegression</code>, a classic and efficient algorithm for binary classification.
</p>

<div className="code-scroll-container mt-2 p-3" style={codeStyle}>
  <code>
    <SyntaxHighlighter language="python" style={darcula}>
{`from sklearn.linear_model import LogisticRegression

# Initialize and train the model
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)

# Predict on the test set
y_pred = logreg.predict(X_test)`}
</SyntaxHighlighter>
  </code>
</div>
<hr style={{ borderTop: "5px solid rgb(241, 239, 239)", margin: "20px 0" }} />
<h3>Step 2.2 – Evaluate Model Performance</h3>
<p>Let’s generate a classification report to measure:</p>
<ul>
  <li><strong>Accuracy</strong> – Overall correct predictions</li>
  <li><strong>Precision</strong> – Of predicted positives, how many were correct?</li>
  <li><strong>Recall</strong> – Of actual positives, how many did you catch?</li>
  <li><strong>F1 Score</strong> – Harmonic mean of precision and recall</li>
</ul>

<div className="code-scroll-container mt-2 p-3" style={codeStyle}>
  <code>
    <SyntaxHighlighter language="python" style={darcula}>
{`from sklearn.metrics import classification_report

# Evaluate the model
report = classification_report(y_test, y_pred, output_dict=True)
print(classification_report(y_test, y_pred))`}
</SyntaxHighlighter>
  </code>
</div>
<hr style={{ borderTop: "5px solid rgb(241, 239, 239)", margin: "20px 0" }} />
<h3>Step 2.3 – Save Performance Metrics</h3>
<p>Saving metrics in a structured format lets you compare multiple models later:</p>

<div className="code-scroll-container mt-2 p-3" style={codeStyle}>
  <code>
    <SyntaxHighlighter language="python" style={darcula}>
{`# Store baseline results
baseline_results = {
    'model': 'Logistic Regression',
    'accuracy': report['accuracy'],
    'precision': report['1']['precision'],
    'recall': report['1']['recall'],
    'f1_score': report['1']['f1-score']
}

print(baseline_results)`}
</SyntaxHighlighter>
  </code>
</div>
<hr style={{ borderTop: "5px solid rgb(241, 239, 239)", margin: "20px 0" }} />
<h3>Notes</h3>
<ul>
  <li>Logistic regression is a linear model, great for interpretability and fast training.</li>
  <li>It sets expectations: if later models only beat it by 1%, the added complexity might not be worth it.</li>
  <li>
    Performance on class <code>1</code> (usually the positive class) is highlighted because it often matters more in
    real-world problems like fraud detection or disease diagnosis.
  </li>
</ul>
<hr style={{ borderTop: "5px solid rgb(241, 239, 239)", margin: "20px 0" }} />
<h3>Analysis</h3>
<p>Think of this like setting your first lap time on a racetrack:</p>
<ul>
  <li>You’re not aiming to win yet — you’re figuring out the course.</li>
  <li>You establish a baseline so you can measure improvements.</li>
  <li>If your first lap is already decent, maybe you don’t need a race car at all.</li>
</ul>
<hr style={{ borderTop: "5px solid rgb(241, 239, 239)", margin: "20px 0" }} />
<h3>Comprehension Check</h3>
<ul>
  <li>What is a baseline model, and why is it useful?</li>
  <li>What does precision tell you that accuracy doesn’t?</li>
  <li>Why is it helpful to save your evaluation metrics programmatically?</li>
</ul>
<hr style={{ borderTop: "5px solid rgb(241, 239, 239)", margin: "20px 0" }} />
<p>Answer the comprehension questions to lock in what you’ve learned!</p>

<hr style={{ borderTop: "5px solid rgb(241, 239, 239)", margin: "20px 0" }} />

    </>

),
},
{
  title: "",
  content: (
    <>
    <h2>Step 3: Train and Compare Alternative Models</h2>
<p>
  You’ve built your baseline, great start! Now it’s time to test out two powerful alternatives: a Support Vector Machine (SVM)
  and a Random Forest. In this step, you’ll train both models, evaluate their performance, and store the results for side-by-side
  comparison with your logistic regression baseline.
</p>
<hr style={{ borderTop: "5px solid rgb(241, 239, 239)", margin: "20px 0" }} />
<h3>Why This Step Is Important</h3>
<p>Trying multiple models helps you:</p>
<ul>
  <li>Experiment with different learning strategies (linear boundary vs. ensemble learning)</li>
  <li>Compare trade-offs in accuracy, overfitting, and runtime</li>
  <li>Make evidence-based decisions about which model to deploy</li>
</ul>
<p>No one model is best for every problem—this is your chance to explore.</p>
<hr style={{ borderTop: "5px solid rgb(241, 239, 239)", margin: "20px 0" }} />
<h3>Step 3.1 – Train a Support Vector Machine (SVM)</h3>
<div className="code-scroll-container mt-2 p-3" style={codeStyle}>
  <code>
    <SyntaxHighlighter language="python" style={darcula}>
{`from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Train the SVM model
svm_model = SVC()
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

# Evaluate
svm_report = classification_report(y_test, y_pred_svm, output_dict=True)
print("SVM Results:")
print(classification_report(y_test, y_pred_svm))

# Store metrics
svm_results = {
    'model': 'Support Vector Machine',
    'accuracy': svm_report['accuracy'],
    'precision': svm_report['1']['precision'],
    'recall': svm_report['1']['recall'],
    'f1_score': svm_report['1']['f1-score']
}`}
</SyntaxHighlighter>
  </code>
</div>
<hr style={{ borderTop: "5px solid rgb(241, 239, 239)", margin: "20px 0" }} />
<h3>Step 3.2 – Train a Random Forest Classifier</h3>
<div className="code-scroll-container mt-2 p-3" style={codeStyle}>
  <code>
    <SyntaxHighlighter language="python" style={darcula}>
{`from sklearn.ensemble import RandomForestClassifier

# Train the random forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Evaluate
rf_report = classification_report(y_test, y_pred_rf, output_dict=True)
print("Random Forest Results:")
print(classification_report(y_test, y_pred_rf))

# Store metrics
rf_results = {
    'model': 'Random Forest',
    'accuracy': rf_report['accuracy'],
    'precision': rf_report['1']['precision'],
    'recall': rf_report['1']['recall'],
    'f1_score': rf_report['1']['f1-score']
}`}
</SyntaxHighlighter>
  </code>
</div>
<hr style={{ borderTop: "5px solid rgb(241, 239, 239)", margin: "20px 0" }} />
<h3>Step 3.3 – Compare Models in a Summary Table</h3>
<p>This table gives you a clean overview of how each model performed, helping you start to think critically about strengths and weaknesses.</p>
<div className="code-scroll-container mt-2 p-3" style={codeStyle}>
  <code>
    <SyntaxHighlighter language="python" style={darcula}>
{`import pandas as pd

# Combine all results
all_results = pd.DataFrame([baseline_results, svm_results, rf_results])
print(all_results)`}
</SyntaxHighlighter>
  </code>
</div>
<hr style={{ borderTop: "5px solid rgb(241, 239, 239)", margin: "20px 0" }} />
<h3>Notes</h3>
<table className="table table-bordered mt-3">
  <thead className="table-light">
    <tr>
      <th>Model</th>
      <th>Strengths</th>
      <th>Weaknesses</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Logistic Regression</td>
      <td>Fast, interpretable</td>
      <td>Limited to linear relationships</td>
    </tr>
    <tr>
      <td>SVM</td>
      <td>Effective in high dimensions</td>
      <td>Slower, harder to tune</td>
    </tr>
    <tr>
      <td>Random Forest</td>
      <td>Powerful, handles non-linearity well</td>
      <td>Less interpretable, larger size</td>
    </tr>
  </tbody>
</table>
<hr style={{ borderTop: "5px solid rgb(241, 239, 239)", margin: "20px 0" }} />
<h3>Analysis</h3>
<p>This is like test driving three vehicles for different terrains:</p>
<ul>
  <li>Logistic regression = compact city car</li>
  <li>SVM = precision motorcycle</li>
  <li>Random forest = rugged all-terrain machine</li>
</ul>
<p>
  Each has its ideal use case. The goal is not to crown a winner, but to match the right tool to the job.
</p>
<hr style={{ borderTop: "5px solid rgb(241, 239, 239)", margin: "20px 0" }} />
<h3>Comprehension Check</h3>
<ul>
  <li>Which model performed best on recall? Why might that matter?</li>
  <li>Did the random forest overfit or generalize well?</li>
  <li>What trade-offs do you see between performance and interpretability?</li>
</ul>
<p>Answer the comprehension questions to lock in what you’ve learned!</p>

<hr style={{ borderTop: "5px solid rgb(241, 239, 239)", margin: "20px 0" }} />

    </>
  ),
},

{  title: "",
  content: (      
    <>
    <h2>Step 4: Visualize and Analyze Errors</h2>
<p>
  Training and evaluating models is only part of the job. Now it’s time to look deeper. In this step, you’ll use confusion matrices to visualize model errors, compare false positives and false negatives, and start to understand the <i>"why"</i> behind the mistakes.
</p>
<hr style={{ borderTop: "5px solid rgb(241, 239, 239)", margin: "20px 0" }} />
<h3>Why This Step Is Important</h3>
<p>Model metrics like accuracy and F1 score are helpful, but they don’t tell you where your model is getting things wrong, or how often.</p>
<p>Here’s what you’ll uncover:</p>
<ul>
  <li>Which types of errors (false positives vs false negatives) are most common</li>
  <li>Whether specific classes or feature patterns are causing problems</li>
  <li>Opportunities for improving data quality or model tuning</li>
</ul>
<p>Good ML practitioners don’t just look at how well a model performs—they look at where it fails and why.</p>
<hr style={{ borderTop: "5px solid rgb(241, 239, 239)", margin: "20px 0" }} />
<h3>Step 4.1 – Generate Confusion Matrices</h3>
<p>A confusion matrix shows:</p>
<table className="table table-bordered w-auto">
  <thead>
    <tr>
      <th></th>
      <th>Predicted 0</th>
      <th>Predicted 1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Actual 0</th>
      <td>True Neg</td>
      <td>False Pos</td>
    </tr>
    <tr>
      <th>Actual 1</th>
      <td>False Neg</td>
      <td>True Pos</td>
    </tr>
  </tbody>
</table>
<hr style={{ borderTop: "5px solid rgb(241, 239, 239)", margin: "20px 0" }} />
<h3>Logistic Regression</h3>
<div className="code-scroll-container mt-2 p-3" style={codeStyle}>
  <code>
    <SyntaxHighlighter language="python" style={darcula}>
{`from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def plot_conf_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Plot for logistic regression
plot_conf_matrix(y_test, y_pred, "Logistic Regression Confusion Matrix")`}
</SyntaxHighlighter>
  </code>
</div>


<h3>SVM and Random Forest</h3>
<div className="code-scroll-container mt-2 p-3" style={codeStyle}>
  <code>
    <SyntaxHighlighter language="python" style={darcula}>
{`# SVM
plot_conf_matrix(y_test, y_pred_svm, "SVM Confusion Matrix")

# Random Forest
plot_conf_matrix(y_test, y_pred_rf, "Random Forest Confusion Matrix")`}
</SyntaxHighlighter>
  </code>
</div>

<h3>Step 4.2 – Analyze False Positives and False Negatives</h3>
<p>You’ll explore where and how your models are misclassifying examples:</p>
<div className="code-scroll-container mt-2 p-3" style={codeStyle}>
  <code>
    <SyntaxHighlighter language="python" style={darcula}>
{`# Example: Find false negatives in Random Forest
false_negatives = (y_test == 1) & (y_pred_rf == 0)
print("False negatives (actual=1, predicted=0):")
print(X_test[false_negatives])`}
</SyntaxHighlighter>
  </code>
</div>

<h3>Step 4.3 – Visualize Feature Ranges of Misclassified Samples</h3>
<p>This step is optional but powerful. Try plotting feature distributions of misclassified vs correctly classified examples:</p>
<div className="code-scroll-container mt-2 p-3" style={codeStyle}>
  <code>
    <SyntaxHighlighter language="python" style={darcula}>
{`import pandas as pd

# Combine features and predictions into a DataFrame
df_test = pd.DataFrame(X_test, columns=[f"feature_{i}" for i in range(X_test.shape[1])])
df_test['true_label'] = y_test
df_test['pred_rf'] = y_pred_rf
df_test['error_type'] = np.where(df_test['true_label'] == df_test['pred_rf'], 'Correct',
                                 np.where(df_test['true_label'] == 1, 'False Negative', 'False Positive'))

# Boxplot for a key feature (e.g., feature_3)
sns.boxplot(x='error_type', y='feature_3', data=df_test)
plt.title("Feature 3 Distribution by Prediction Outcome")
plt.show()`}
</SyntaxHighlighter>
  </code>
</div>

<h3>Notes</h3>
<ul>
  <li>Confusion matrices highlight asymmetries in how your model behaves.</li>
  <li>False negatives might be more costly in some domains (e.g., missed fraud or cancer diagnoses).</li>
  <li>Misclassifications often reveal data quality issues or places where feature engineering could help.</li>
</ul>

<h3>Analysis</h3>
<p>This is like reviewing replays of a game:</p>
<ul>
  <li>You’re not just looking at the final score—you’re replaying the moments where things went wrong.</li>
  <li>This helps you see why you missed the goal or made a bad pass, so you can do better next time.</li>
</ul>

<h3>Comprehension Check</h3>
<ul>
  <li>What’s the difference between a false positive and false negative?</li>
  <li>Which type of error would be worse in your application—and why?</li>
  <li>Do certain features appear more often in misclassified samples?</li>
</ul>
<p>Answer the comprehension questions to lock in what you’ve learned!</p>

<hr style={{ borderTop: "5px solid rgb(241, 239, 239)", margin: "20px 0" }} />

    </>
  ),
},
{
  title: "",
  content: (
    <>
    <h2>Step 5: Recommend Improvements and Next Steps</h2>
<p>
  You’ve built, tested, and analyzed multiple models—great work! Now it’s time to reflect on what you’ve learned, interpret your findings, and plan your next move. This is a critical (and often skipped) step in real-world machine learning projects: turning analysis into actionable insights.
</p>

<h3>Why This Step Is Important</h3>
<p>Building models is just the beginning. In practice, you need to:</p>
<ul>
  <li>Optimize and iterate to improve performance</li>
  <li>Understand trade offs (e.g., complexity vs. interpretability)</li>
  <li>Adjust your pipeline based on what you've learned</li>
</ul>
<p>ML is an iterative process—your first model is rarely your last.</p>
<hr style={{ borderTop: "4px solid rgb(241, 239, 239)", margin: "20px 0" }} />
<h3>Step 5.1 – Review Performance Summary</h3>
<p>Let’s bring everything together and look at the comparison table:</p>
<div className="code-scroll-container mt-2 p-3" style={codeStyle}>
  <code>
    <SyntaxHighlighter language="python" style={darcula}>
{`# Display all collected results
print(all_results.sort_values(by='f1_score', ascending=False))`}
</SyntaxHighlighter>
  </code>
</div>
<hr style={{ borderTop: "4px solid rgb(241, 239, 239)", margin: "20px 0" }} />
<p>Ask yourself:</p>
<ul>
  <li>Which model had the best F1 score?</li>
  <li>Were precision and recall balanced, or skewed toward one?</li>
  <li>Did a more complex model (like Random Forest) significantly outperform the baseline?</li>
</ul>
<hr style={{ borderTop: "4px solid rgb(241, 239, 239)", margin: "20px 0" }} />
<h3>Step 5.2 – Interpret the Results</h3>
<p>Here are some prompts to guide interpretation:</p>
<ul>
  <li>Did the ensemble model (Random Forest) outperform the simpler models? Why might that be?</li>
  <li>Was the SVM slower or harder to tune? Did it offer any benefits in accuracy?</li>
  <li>Did logistic regression hold its own in terms of speed and interpretability?</li>
</ul>
<p>
  Consider which model you’d actually deploy, not just based on metrics, but also based on practical constraints like runtime and explainability.
</p>
<hr style={{ borderTop: "4px solid rgb(241, 239, 239)", margin: "20px 0" }} />
<h3>Step 5.3 – Identify Improvement Opportunities</h3>
<p>Based on your confusion matrices and feature exploration in Step 4, think about what could make your models better.</p>
<hr style={{ borderTop: "4px solid rgb(241, 239, 239)", margin: "20px 0" }} />
<h3>Some ideas:</h3>
<b>Modeling Techniques</b>
<ul>
  <li>Perform hyperparameter tuning (e.g., grid search on SVM or Random Forest)</li>
  <li>Try regularization in logistic regression</li>
  <li>Experiment with nonlinear kernels for SVM</li>
</ul>

<b>Feature Engineering</b>
<ul>
  <li>Use feature selection to drop low value features</li>
  <li>Apply Principal Component Analysis (PCA) for dimensionality reduction</li>
  <li>Engineer new features based on domain knowledge</li>
</ul>

<b>Handling Imbalanced Data</b>
<ul>
  <li>Use stratified sampling during train/test split</li>
  <li>Apply resampling techniques (SMOTE, undersampling, etc.)</li>
  <li>Adjust class weights in the model</li>
</ul>
<hr style={{ borderTop: "4px solid rgb(241, 239, 239)", margin: "20px 0" }} />
<h3>Step 5.4 – Plan Your Next Iteration</h3>
<p>Choose 1–2 next steps and outline what you would do differently:</p>
<b>Next Steps:</b>
<ul>
  <li>Perform GridSearchCV on Random Forest to tune <code>n_estimators</code> and <code>max_depth</code></li>
  <li>Try PCA to reduce feature set and improve generalization</li>
  <li>Adjust class weights in SVM to improve recall on minority class</li>
</ul>
<p>
  Keep it focused and measurable—improvement plans should be iterative, not all at once.
</p>
<hr style={{ borderTop: "4px solid rgb(241, 239, 239)", margin: "20px 0" }} />
<h3>Analysis</h3>
<p>This is like watching game film with your coach:</p>
<ul>
  <li>You’ve played the match (trained the model).</li>
  <li>You’ve reviewed the highlights and mistakes (error analysis).</li>
  <li>Now you and the coach decide how to train smarter next week.</li>
</ul>
<p>This is where you level up your modeling strategy.</p>
<hr style={{ borderTop: "4px solid rgb(241, 239, 239)", margin: "20px 0" }} />
<h3>Comprehension Check</h3>
<ul>
  <li>What did you learn from the error patterns?</li>
  <li>Which model would you recommend and why?</li>
  <li>What’s one concrete thing you would change before your next training run?</li>
</ul>
<p>Answer the comprehension question to lock in what you’ve learned!</p>

<hr style={{ borderTop: "5px solid rgb(241, 239, 239)", margin: "20px 0" }} />

<h3>🎉 You’ve Completed the Lab!</h3>
<p>Nice work! You’ve:</p>
<ul>
  <li>Explored a real dataset</li>
  <li>Built a baseline model</li>
  <li>Trained and compared alternatives</li>
  <li>Analyzed errors in detail</li>
  <li>Proposed meaningful improvements</li>
</ul>
<p>You now have a repeatable workflow for building and evaluating machine learning models.</p>

    </>
  ),
},
]