import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1

def load_file(filename='online_retail.xlsx'):
    df = pd.read_csv(filename,low_memory=False)
    return df

def preprocessing(df):
    df['Description'] = df['Description'].str.strip()
    df.dropna(axis=0, subset=['InvoiceNo'], inplace=True)
    df['InvoiceNo'] = df['InvoiceNo'].astype('str')
    df = df[~df['InvoiceNo'].str.contains('C')]
    return df


def generate_basket(df, country="France",min_support=0.07,max_length=3):
    basket = (df[df['Country'] == country ]
          .groupby(['InvoiceNo', 'Description'])['Quantity']
          .sum().unstack().reset_index().fillna(0)
          .set_index('InvoiceNo'))
    basket_sets = basket.applymap(encode_units)
    try:basket_sets.drop('POSTAGE', inplace=True, axis=1)
    except:pass
    frequent_itemsets = apriori(basket_sets, min_support=min_support , use_colnames=True,max_len=max_length)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
    return rules

def get_rules(rules, lift= 6, confidence= 0.8):
    result = rules[ (rules['lift'] >= lift) & (rules['confidence'] >= confidence) ]
    return result



if __name__ == "__main__":
    print("loading data")
    df = load_file('masket_basket_optimization\static\datasets\cleaned_online_retail.csv') # load
    print("cleaning data")
    df = preprocessing(df) # clean it up
    print(df['Country'].unique().tolist())
    print("generating basket optimised recommedations")
    rules = generate_basket(df)
    print("loading rules",rules)
    results = get_rules(rules)
    print("RESULTS")
    print(results)