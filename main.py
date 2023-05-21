import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression

data = pd.read_csv('./oecd_bli_pivoted.csv')

def barChart():
    selected_data = data[['Country', 'Life satisfaction']]
    sorted_data = selected_data.sort_values(by='Life satisfaction', ascending=False)
    plt.figure(figsize=(10, 6))
    plt.bar(sorted_data['Country'], sorted_data['Life satisfaction'])
    plt.xlabel('Negara')
    plt.ylabel('Life Satisfaction')
    plt.title('Perbandingan Life Satisfaction antara Negara')
    plt.xticks(rotation=90)
    plt.show()


def scatterChart():
    selected_data = data[['Employment rate', 'Life satisfaction']]
    plt.figure(figsize=(10, 6))
    plt.scatter(selected_data['Employment rate'], selected_data['Life satisfaction'])
    plt.xlabel('Employment Rate')
    plt.ylabel('Life Satisfaction')
    plt.title('Hubungan antara Employment Rate dan Life Satisfaction')
    plt.show()


def boxChart():
    plt.figure(figsize=(10, 6))
    data.boxplot(column='Life satisfaction', by='Educational attainment', vert=False)
    plt.xlabel('Life Satisfaction')
    plt.ylabel('Educational Attainment')
    plt.title('Box Plot: Tingkat Pendidikan dan Life Satisfaction')
    plt.show()


def dataDescriptive():
    print(data.info())
    desc = data.describe()
    print(desc)
    desc.to_csv('./dataDescribe.csv')
    print(data['Country'].unique())
    print("Jumlah Negara:", len(data['Country'].unique()))
    print("Jumlah Baris dan Kolom:", data.shape)


def lifeSatisfactoryAttribute():
    significant_attributes = []

    for column in data.columns:
        if column != 'Life satisfaction' and column != 'Country':
            stat, p_value = stats.ttest_ind(data[column], data['Life satisfaction'])

            # Menerapkan tingkat kepercayaan 90% (p-value < 0.1)
            if p_value < 0.1:
                significant_attributes.append(column)

    print("Atribut yang mempengaruhi Life Satisfaction:")
    print(significant_attributes)


def lifeSatisfactoryRegression():
    data1 = data.dropna()
    y = data1['Life satisfaction']
    X = data1.drop(columns=['Life satisfaction', 'Country'])

    model = LinearRegression()
    model.fit(X, y)

    coefficients = pd.DataFrame({'Attribute': X.columns, 'Coefficient': model.coef_})
    print(coefficients)

    significant_attributes = coefficients[coefficients['Coefficient'] != 0]['Attribute']
    print("Atribut yang signifikan terhadap Life Satisfaction:")
    print(significant_attributes)


if __name__ == '__main__':
    barChart()
    scatterChart()
    boxChart()
    dataDescriptive()
    lifeSatisfactoryAttribute()
    lifeSatisfactoryRegression()
