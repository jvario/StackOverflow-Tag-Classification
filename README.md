# StackOverflow-Tag-Classification

# I.  Introduction:

#### Project Overview:
The initial development phase of the code base involves implementing two models for tag prediction on Stack Overflow questions:

a) Baseline Tag Prediction Model with Classic Classifiers:

  - Data Preprocessing: Cleaning and preprocessing Stack Overflow question data from provided datasets.
  - Feature Engineering: Extracting relevant features for classic classifiers.
  - Model Selection: Choosing classic classifiers like Logistic Regression, Naive Bayes, etc.
  - Model Training: Training classifiers and evaluating performance metrics.
    
b) Sophisticated Model with a Language Model Backbone:

  - Data Preparation: Preparing Stack Overflow question data for input to Language Model.
  - LM Selection: Choosing a suitable LM architecture like BERT, GPT, etc.
  - Fine-tuning: Fine-tuning the selected LM on Stack Overflow data.
  - Training and Evaluation: Training and evaluating the model's performance.
  - Comparison: Comparing performance with the baseline model.


#### Dataset - StackSample: 10% of Stack Overflow Q&A:
Dataset with the text of 10% of the entire Stack Overflow question and answer corpus. Specifically, we're only using two files from this dataset:

- Questions.csv : This file includes unique identifiers for the users who created each question, as well as for the questions themselves. It also provides timestamps for when each question was created and closed, along with the cumulative reaction score (positive, negative, or zero) for each question. Additionally, it contains the title and main body of each question.

- Tags.csv: This file contains unique identifiers for each question, along with one or more associated tags.

You can download the dataset from [here](https://www.kaggle.com/datasets/stackoverflow/stacksample/data).

### Installation


    # go to your home dir
    git clone https://github.com/jvario/StackOverflow-Tag-Classification.git
    cd StackOverflow-Tag-Classification

    # build the image
    docker build -t docker-tag-clf -f Dockerfile . 
    docker run -p 8888:8888 docker-tag-clf
    

# II.  Pipeline:

#### EDA:
After some exploratory data analysis (EDA), discovered that the dataset is very large and requires reduction into a subset to train our models due to memory allocation and limited time constraints. Therefore, decided to focus on creating a smaller, more manageable subset of the data for model training purposes, restricting it to include only the most common tags. It's worth noting that chose to include only questions with a positive cumulative score, as we believe these questions often offer valuable insights and solutions, thus ensuring the inclusion of high-quality data.

#### Preproccess:
In order to perform text sanitization on our data, we applied the following steps:

- Joining
- Lowercase
- Clean/Remove NaN values
- Remove HTML tags
- Remove Panctuation
- Remove StopWords
- Tokenization
- Lemmatization

#### Baseline Tag Prediction Model with Classic Classifiers:
In our multilabel problem scenario, each data point can be associated with multiple tags. This presents a challenge in appropriately partitioning the original dataset. To tackle this, we've devised a function intended for use in both model notebooks. Its primary objective is to effectively divide the data into training and testing sets, and optionally validation sets, with adjustable proportions.

Additionally, for feature extraction, we've applied **TF-IDF (Term Frequency-Inverse Document Frequency)**. This technique helps to represent each document in the dataset as a vector based on the importance of each word, considering both its frequency in the document and its rarity across all documents.

Furthermore, we've utilized the **Multilabel Binarizer** to encode the multilabel tags into binary format, facilitating the multilabel classification task.

For evaluating the performance of our models, we've chosen several metrics including **Hamming loss, recall, F1 score, support, Jaccard score, and precision**. These metrics provide insights into different aspects of the model's performance, such as its ability to correctly classify each label, handle imbalanced data, and capture the trade-off between precision and recall.

#### LLM:

For the Long-Text Multilabel (LLM) task, we chose the BERT model. Due to GPU limitations, we ran the BERT model training and evaluation in Google Colab. Utilizing BERT for text classification, we leveraged its deep contextualized representations. To handle the multilabel nature of the task, have been used **Multilabel Binarizer** to encode tags into binary format. For detailed evaluation, have been used ***Classification Report**, providing **precision**, **recall**, **F1-score**  for each label. Additionally, we assessed model performance using **Hamming loss**, **accuracy**, and **loss metrics**. Furthermore, we calculated the **Jaccard score** to measure the similarity between predicted and true labels.

# III.  Results:

| Model | Sample Size | Accuracy | Jaccard Score | Hamming Loss |
|-------|-------------|----------|---------------|--------------|
| SVC   | ~44000        | 0.56     | 0.58          | 0.03       |
| SGD   | ~44000        | 0.52     | 0.52          | 0.04       |
| BERT  | ~44000        | 0.77     | 0.67          | 0.03       |

This table represents the evaluation results for different models based on question **scores** **greater than 5**, filtered for the **most common** **15 tags**. The metrics include accuracy, Jaccard score, and Hamming loss.

# IV. Conclusion:
Based on the evaluation results presented in the table, it is evident that the BERT model outperforms both the SVC and SGD models in terms of accuracy and Jaccard score. With an accuracy of 0.77 and a Jaccard score of 0.67, BERT demonstrates superior performance compared to SVC and SGD, which have accuracies of 0.56 and 0.52, and Jaccard scores of 0.58 and 0.52, respectively. Additionally, all three models exhibit low Hamming loss, indicating efficient performance in multi-label classification tasks. These findings suggest that BERT is a promising model for accurately predicting tags based on question scores greater than 5, particularly when considering the most common 15 tags. Furthermore, it's essential to acknowledge that with sufficient resources and adequate time allocation, there is potential for even better results and more in-depth investigation. Given the constraints faced in this study, including limited computational resources and time, further exploration and optimization could yield improved model performance and deeper insights into the dataset. It's worth noting that the dataset used for training and evaluation was substantial, requiring significant computational resources. Despite this limitation, the results obtained provide valuable insights into the performance of the models under consideration.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
