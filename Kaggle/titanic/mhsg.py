import pandas as ps
import sklearn.linear_model as lrm
from sklearn.preprocessing import Imputer
from sklearn.feature_extraction import DictVectorizer

# Read about ghosl in Islam :D
def ghosl(dirty_df,vectorizer):
    # Vectorizing
    dirty_df = dirty_df.to_dict(orient='records')
    cleaned_df = vectorizer.transform(dirty_df)

    # Remove NaN
    imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
    imp.fit(cleaned_df)
    cleaned_df_imputed = imp.transform(cleaned_df)

    return cleaned_df_imputed


def show_result(cleaned_features, cleaned_target, model):
    # Make sure you've already ghosled your data
    SAMPLE = len(cleaned_features)
    falses = abs((model.predict(cleaned_features[:SAMPLE]) - cleaned_target.head(SAMPLE).values)).sum()

    return (100 - (falses / SAMPLE) * 100)

def solve(features,model, vectorizer):
    ghosled_features = ghosl(features,vectorizer)
    result = model.predict(ghosled_features)
    return result


def feature_cleaning(features,survived_removal=True):
    if survived_removal:
        features = features.drop(['Survived'], axis=1)

    features = features.drop(['PassengerId'], axis=1)
    features = features.drop(['Name'], axis=1)
    features = features.drop(['Ticket'], axis=1)
    features = features.drop(['Cabin'], axis=1)

    return features

# Splitting data to training and testing
train_set_ration = 0.99
titan = ps.read_csv('data/train.csv')
train_set_size = int(train_set_ration * len(titan))
test_set_size = len(titan) - train_set_size

train_set = titan.loc[:train_set_size]
test_set = titan.loc[train_set_size:]

# Create features and target for train
train_features = feature_cleaning(train_set)
train_target = train_set['Survived']

test_features = feature_cleaning(test_set)
test_target = test_set['Survived']

# Pre-processing
vectorizer = DictVectorizer (sparse = False)
vectorizer.fit(train_features.to_dict(orient = 'records'))

cleaned_train_features = ghosl(train_features, vectorizer)
cleaned_test_features = ghosl(test_features, vectorizer)

# Make model (Logistic regression)
lr = lrm.LogisticRegression()
lr.fit(cleaned_train_features,train_target)

print("Train set accuracy: " + str (show_result(cleaned_train_features,train_target,lr)))
print("Test set accuracy: " + str(show_result(cleaned_test_features, test_target, lr)))

eval_data = ps.read_csv('data/test.csv')
cleaned_eval_data = feature_cleaning(eval_data,False)
result = solve(cleaned_eval_data,lr,vectorizer)

with open('output2.csv','w') as file:
    file.write('PassengerId,Survived')
    index = 892
    for x in result:
        file.write('\n' + str(index) + "," + str(x))
        index += 1